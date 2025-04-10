from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import json
from abc import ABC, abstractmethod

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."
logger = Logger()


class Product():
    def __init__(self, product: str, limit: int, state: TradingState):
        self.traderData = state.traderData
        self.timestamp = state.timestamp
        self.listings = state.listings
        self.order_depth = state.order_depths[product]
        self.own_trades = state.own_trades
        self.market_trades = state.market_trades
        self.position = state.position
        self.observations = state.observations
        self.product = product
        self.limit = limit
        self.position = state.position[product] if product in state.position else 0
        self.nsell = 0
        self.nbuy = 0
        self.orders: List[Order] = []
    
    def buy(self, price: int, quantity: int):
        self.orders.append(Order(self.product, int(price), quantity))
        self.nbuy += quantity
    def sell(self, price: int, quantity: int):
        self.orders.append(Order(self.product, int(price), -quantity))
        self.nsell += quantity
    def max_buy_orders(self):
        return self.limit - self.position - self.nbuy 
    def max_sell_orders(self):
        return self.limit + self.position - self.nsell
           
    def market_take(self, bid_p_lim: int, ask_p_lim: int):
        logger.print("Market Taking:")
        for i in range(len(self.order_depth.buy_orders)):
            bid_price, bid_vol = list(self.order_depth.buy_orders.items())[i]
            if bid_price > bid_p_lim: 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if sell_vol == 0:
                    break

                logger.print(f"Market take: selling {sell_vol}@{bid_price}\n")
                self.sell(bid_price, sell_vol)
            
            if self.position - self.nsell <= -self.limit:
                break

        for i in range(len(self.order_depth.sell_orders)):
            ask_price, ask_vol = list(self.order_depth.sell_orders.items())[i]
            ask_vol *= -1
            if ask_price < ask_p_lim:
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if buy_vol == 0:
                    break
                logger.print(f"Market take: buying {buy_vol}@{ask_price}\n")
                self.buy(ask_price, buy_vol)
            
            if self.position + self.nbuy >= self.limit:
                break

    
    def neutralize(self, fv: int, agg_up: int, agg_down: int, aggressive=False):
        logger.print("Neutralizing position")
        new_position = self.position + self.nbuy - self.nsell
        if new_position > 0:
            if aggressive:
                sell_vol = min(new_position, self.max_sell_orders())
                self.sell(fv + agg_up, sell_vol)
            else:
                if len(self.order_depth.buy_orders) > 0:
                    best_bid_price, best_bid_vol = max(self.order_depth.buy_orders.items(), key=lambda x: x[0], default=(fv, new_position))

                    if best_bid_price >= fv:
                        sell_vol = min(min(new_position, best_bid_vol), self.max_sell_orders()) # additional param of new_position to get to net neutral
                        logger.print(f"Neutralizing {new_position} to {new_position - sell_vol} at {best_bid_price}\n")
                        self.sell(best_bid_price, sell_vol)

        elif new_position < 0:
            if aggressive: # buy as much as possible to get to 0 without overflowing buy orders
                buy_vol = min(-new_position, self.max_buy_orders())
                self.buy(fv - agg_down, buy_vol)
            else:
                if len(self.order_depth.sell_orders) > 0:
                    best_ask_price, best_ask_vol = max(self.order_depth.sell_orders.items(), key = lambda x: x[0], default=(fv, new_position))
                    if best_ask_price <= fv:
                        buy_vol = min(min(-new_position, -best_ask_vol), self.max_sell_orders())
                        logger.print(f"Neutralizing {new_position} to {new_position + buy_vol} at {best_ask_price}")
                        self.buy(best_ask_price, buy_vol)


    def market_make(
        self,
        buy_price: int,
        sell_price: int
    ):
        self.buy(buy_price, self.max_buy_orders())
        self.sell(sell_price, self.max_sell_orders())
    
    def market_make_undercut(
        self,
        fair_val: int,
        edge: float,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1

        return self.market_make(mm_buy, mm_sell)
    
    def strategy(self): # RUNTIME POLYMORPHISM BTW
        raise NotImplementedError()
        ...

    def execute(self): 
        self.strategy()
        return self.orders, self.traderData

class Resin(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv, fv)
        self.neutralize(fv, 1, 1, False)
        self.market_make_undercut(fv, 1)


class Kelp(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        mm_bid_price, mm_bid_qty = max(self.order_depth.buy_orders.items(), key = lambda x: x[1])
        mm_ask_price, mm_ask_qty = max(self.order_depth.sell_orders.items(), key = lambda x: x[1])
        return (mm_bid_price + mm_ask_price)/2
    
    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv, fv)
        self.neutralize(fv, 1, 1, False)
        self.market_make_undercut(fv, 1)

class Ink(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
        self.state = state

    def fair_val(self):
        mm_bid_price, mm_bid_qty = max(self.order_depth.buy_orders.items(), key = lambda x: x[0])
        mm_ask_price, mm_ask_qty = min(self.order_depth.sell_orders.items(), key = lambda x: x[0])
        
        mid = (mm_bid_price + mm_ask_price)/2
        prev_mid = self.state.traderData#.split(" ")[-1]
        if prev_mid == "":
            prev_mid = mid

        self.state.traderData = self.state.traderData + " " + str(mid)
        
        val = mid * 0.9 + prev_mid * 0.1
        return val
    
    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv, fv)
        self.neutralize(fv, 1, 1, True)
        self.market_make_undercut(fv, 2)

class Trader:
    def executor(self, product: str, state: TradingState): # add product to execute set of strategies

        if product == "RAINFOREST_RESIN":
            resin = Resin(product, 50, state)
            return resin.execute()
            
        if product == "KELP":
            kelp = Kelp(product, 50, state)
            return kelp.execute()
        
        if product == "SQUID_INK":
            ink = Ink(product, 50, state)
            return ink.execute()
        
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        result = {}

        traderData = ""
        for product in state.order_depths:
            position = state.position[product] if product in state.position else 0

            result[product], data_prod = self.executor(product, state)          
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
