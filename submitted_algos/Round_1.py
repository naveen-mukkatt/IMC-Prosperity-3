from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import json
import math
import queue
import numpy as np
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
        self.hist_mid: List[float] = []
        self.hist_mm_mid: List[float]= []
        self.init_hist_mid()
        self.orders: List[Order] = []
        self.window = 10
    
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
    def active_position(self):
        return self.position + self.nbuy - self.nsell
    
    def best_bid(self):
        if len(self.order_depth.buy_orders) > 0:
            return max(self.order_depth.buy_orders.keys())
        else:
            return math.nan
        
    def best_ask(self):
        if len(self.order_depth.sell_orders) > 0:
            return min(self.order_depth.sell_orders.keys())
        else:
            return math.nan

    def mid_price(self):
        return (self.best_bid() + self.best_ask()) / 2
        
    def market_take(self, fair_val: float):
        logger.print("Market Taking:")
        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > fair_val or (bid_price == fair_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == fair_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    logger.print(f"Market take: selling {sell_vol}@{bid_price}\n")
                    self.sell(bid_price, sell_vol)


        for ask_price, ask_vol in self.order_depth.sell_orders.items():
            ask_vol *= -1
            if ask_price < fair_val or (ask_price == fair_val and self.active_position() < 0):
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if ask_price == fair_val:
                    buy_vol = min(buy_vol, -self.active_position())
                if buy_vol > 0:
                    logger.print(f"Market take: buying {buy_vol}@{ask_price}\n")
                    self.buy(ask_price, buy_vol)

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
    
    def max_vol_mid(self):
        mm_bid_price, mm_bid_qty = max(self.order_depth.buy_orders.items(), key = lambda x: x[1])
        mm_ask_price, mm_ask_qty = max(self.order_depth.sell_orders.items(), key = lambda x: x[1])
        return (mm_bid_price + mm_ask_price)/2
    
    def init_hist_mid(self):
        prods = self.traderData.split("\n")
        ind = -1
        for i in range(len(prods)):
            arr = prods[i].split(" ")
            if arr[0] == self.product:
                ind = i
                break
        
        if ind >= 0:
            for price in prods[ind].split(" ")[1:]:
                self.hist_mid.append(float(price))
            for mm_price in prods[ind + 1].split(" ")[1:]:
                self.hist_mm_mid.append(float(mm_price))

    def return_mids(self):
        """Converts self.hist_mid and self.hist_mm_mid into string form for TraderData.
        String form: [PRODUCT] [mp1] [mp2] ... [mp10]
        [PRODUCT] [mm_mp1] [mm_mp2] ... [mm_mp10]"""
        self.hist_mid.append(self.mid_price())
        self.hist_mm_mid.append(self.max_vol_mid())
        self.hist_mid = self.hist_mid[-self.window:]
        self.hist_mm_mid = self.hist_mm_mid[-self.window:]
        res = self.product
        for price in self.hist_mid:
            res = res + " " + str(price)
        res += "\n" + self.product
        for price in self.hist_mm_mid:
            res = res + " " + str(price)
        return res


    def hist_mid_make(self, mm_bot: bool=False):
        hm = self.hist_mid
        if mm_bot:
            hm = self.hist_mm_mid
        
        if len(hm) > 0:
            value = sum(hm) / len(hm)
        else:
            if mm_bot:
                value = self.max_vol_mid()
            else:
                value = self.mid_price()
        offset = max(1, (self.best_ask() - self.best_bid()) // 2)
        buy_price = round(value - offset)
        sell_price = round(value + offset)
        order_size = min(self.max_buy_orders(), self.max_sell_orders())
        self.buy(buy_price, order_size)
        self.sell(sell_price, order_size)

    def strategy(self): # RUNTIME POLYMORPHISM BTW
        raise NotImplementedError()
        ...

    def execute(self, blank: bool=False): 
        self.strategy()
        if blank:
            return [], ""
        return self.orders, self.return_mids()

class Resin(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv)
        self.market_make_undercut(fv, 1)


class Kelp(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return self.max_vol_mid()

    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv)
        self.market_make_undercut(fv, 1)

class Ink(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
        self.gamma = 0
        self.window = 10

    def mean_hist(self): 
        return sum(self.hist_mid)/len(self.hist_mid) if len(self.hist_mid) > 0 else self.mid_price()
    
    def ou(self):
        # OU adjustment: reversion toward fair value
        adj_price = self.mid_price() + self.gamma * (self.mean_hist() - self.mid_price())

        # Create bid/ask around adjusted price
        spread = 1  # you can tune this
        bid_quote = int(adj_price - spread)
        ask_quote = int(adj_price + spread)
        bid_vol = min(20, self.max_buy_orders())
        ask_vol = min(20, self.max_sell_orders())

        self.buy(bid_quote, bid_vol)
        self.sell(ask_quote, ask_vol)

    def compute_z_score(self):
        if len(self.prices) < self.window:
            return 0
        recent = np.array(self.hist_mid)
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0
        return (self.mid_price() - mean) / std
    
    def fair_val(self):        
        mid = self.max_vol_mid()
        prev_mid = mid
        if len(self.hist_mm_mid) > 0:
            prev_mid = self.hist_mm_mid[-1]

        
        val = mid * 0.9 + prev_mid * 0.1
        return val
    
    def strategy(self):
        #fv = self.fair_val()
        #self.market_take(fv)
        #self.market_make_undercut(fv, 2)
        self.ou()

class Croissant(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    

    def fair_val(self):
        return 0

    def strategy(self):
        ...

class Croissant(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    

    def fair_val(self):
        return 0

    def strategy(self):
        ...
        
class Djembe(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    

    def fair_val(self):
        return 0

    def strategy(self):
        ...
        
class Basket1(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    

    def fair_val(self):
        return 0

    def strategy(self):
        ...

class Basket2(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    

    def fair_val(self):
        return 0

    def strategy(self):
        ...


class Trader:
    def executor(self, product: str, state: TradingState): # add product to execute set of strategies

        if product == "RAINFOREST_RESIN":
            resin = Resin(product, 50, state)
            return resin.execute(blank=False)
            
        if product == "KELP":
            kelp = Kelp(product, 50, state)
            return kelp.execute(blank=False)
        
        if product == "SQUID_INK":
            ink = Ink(product, 50, state)
            return ink.execute(blank=False)
        
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        result = {}

        traderData = ""
        for product in state.order_depths:
            result[product], data_prod = self.executor(product, state)  
            traderData += data_prod + "\n"        
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
