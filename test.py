from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import json

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

RR = "RAINFOREST_RESIN"

class Trader:       
    def market_take(self, product: str, bid_price_limit: int, ask_price_limit: int, position: int, limit: int, order_depth: OrderDepth, orders: list[Order]):
        # make a list of every trade you can take.
        logger.print("Executing Market Take on " + product)
        nbuy = 0
        nsell = 0

        for i in range(len(order_depth.buy_orders)):
            bid_price, bid_vol = list(order_depth.buy_orders.items())[i]
            if bid_price > bid_price_limit: 
                # sell as many as possible here
                sell_vol = min(limit + position - nsell, bid_vol)

                if sell_vol == 0:
                    break

                logger.print(f"Market take: selling {sell_vol}@{bid_price}\n")
                orders.append(Order(product, bid_price, -sell_vol))
                nsell += sell_vol
            
            if position - nsell <= -limit:
                break

        for i in range(len(order_depth.sell_orders)):
            ask_price, ask_vol = list(order_depth.sell_orders.items())[i]
            ask_vol *= -1
            if ask_price < ask_price_limit:
                buy_vol = min(limit - position - nbuy, ask_vol)
                if buy_vol == 0:
                    break
                logger.print(f"Market take: buying {buy_vol}@{ask_price}\n")
                orders.append(Order(product, ask_price, buy_vol))
                nbuy += buy_vol
            
            if position + nbuy >= limit:
                break

        # can try interleaving the orders to get more matches?
        # wait nvm, only one of buy/sell can exist l o l
        return nbuy, nsell
    
    # TODO: this is doing a horrible job of neutralizing. test out aggression and see how it impacts PnL
    # current benchmark: 1.9k on Day 1 Resin
    
    def neutralize(self, product: str, fair_val: int, position: int, nbuy: int, nsell: int, limit: int, order_depth: OrderDepth, orders: list[Order], aggressive=False):
        logger.print("Neutralizing position on " + product)
        new_position = position + nbuy - nsell


        if new_position > 0:
            if aggressive:
                sell_vol = min(new_position, limit - nsell + position)
                orders.append(Order(product, 10001, -sell_vol))
                nsell += sell_vol
            else:
                if len(order_depth.buy_orders) > 0:
                    best_bid_price, best_bid_vol = max(order_depth.buy_orders.items(), key=lambda x: x[0], default=(10000, new_position))

                    if best_bid_price >= fair_val:
                        sell_vol = min(min(new_position, best_bid_vol), limit + new_position - nsell) # additional param of new_position to get to net neutral
                        logger.print(f"Neutralizing {new_position} to {new_position - sell_vol} at {best_bid_price}\n")
                        orders.append(Order(product, best_bid_price, -sell_vol))
                        nsell += sell_vol

        elif new_position < 0:
            if aggressive: # buy as much as possible to get to 0 without overflowing buy orders
                buy_vol = min(-new_position, limit - nbuy - position)
                orders.append(Order(product, 9999, buy_vol))
                nbuy += buy_vol
            else:
                if len(order_depth.sell_orders) > 0:
                    best_ask_price, best_ask_vol = max(order_depth.sell_orders.items(), key = lambda x: x[0], default=(10000, new_position))
                    if best_ask_price <= fair_val:
                        buy_vol = min(min(-new_position, -best_ask_vol), limit - new_position - nbuy)
                        logger.print(f"Neutralizing {new_position} to {new_position + buy_vol} at {best_ask_price}")
                        orders.append(Order(product, best_ask_price, buy_vol))
                        nbuy += buy_vol

        return nbuy, nsell

    def market_make(
        self,
        product: str,
        position: int, 
        buy_price: int, sell_price: int,
        nbuy: int, nsell: int, 
        limit: int, 
        order_depth: OrderDepth, 
        orders: list[Order], 
        slight_undercut: bool=True, 
        bid_target=9999, ask_target=10001
    ):
        logger.print("Starting MM Algorithm: " + product)

        if slight_undercut:    
            ...
        
        buy_qty = limit - (position + nbuy)
        sell_qty = limit - (nsell - position)

        orders.append(Order(product, buy_price, buy_qty))
        orders.append(Order(product, sell_price, -sell_qty))

        nbuy += buy_qty
        nsell += sell_qty

        return nbuy, nsell
    
    def market_make_undercut(
        self,
        product: str,
        position: int, 
        fair_val: int,
        edge: float,
        nbuy: int, nsell: int, 
        limit: int, 
        order_depth: OrderDepth, 
        orders: list[Order], 
        slight_undercut: bool=True,
    ):
        mm_buy = max([bid for bid in order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1
        
    def executor(self, order_depth: OrderDepth, product: str, position: int): # add product to execute set of strategies
        traderData = product
        orders: List[Order] = []

        POS_LIMIT = {"KELP": 50, RR: 50}
        RR_fair_val = 10000
        RR_bid_edge = 1
        RR_ask_edge = 1

        if product == RR:
            lim = POS_LIMIT[product]
            logger.print("Executing RR Strategy")
            nbuy, nsell = self.market_take(product, RR_fair_val, RR_fair_val, position, lim, order_depth, orders)
            logger.print(f"Market taking done. Nbuy = {nbuy}, Nsell = {nsell}\n")
            nbuy, nsell = self.neutralize(product, RR_fair_val, position, nbuy, nsell, lim, order_depth, orders, aggressive=False)
            logger.print(f"Market neutralization done. Nbuy = {nbuy}, Nsell = {nsell}\n")
            nbuy, nsell = self.market_make(product, RR_fair_val, position, RR_bid_edge, RR_ask_edge, nbuy, nsell, lim, order_depth, orders, slight_undercut=True, bid_target=9998, ask_target=10002)
            logger.print(f"Market making done. Nbuy = {nbuy}, Nsell = {nsell}\n")
            logger.print("RR Strategy done.")
            

        
        return orders, traderData


    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        result = {}

        traderData = "ITERATION"

        #for product in state.order_depths:
        #    order_depth: OrderDepth = state.order_depths[product]
        #    position = state.position[product] if product in state.position else 0

        #    result[product], data_prod = self.executor(order_depth, product, position)
        #    traderData = traderData + "\n" + data_prod           
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
