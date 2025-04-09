# Best result: 1.9k on IMC test, 3.5k on Jasper Day 0, 51.1k on Jasper Day 1. (All Resin)

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

    def identify_mm_bot(self, order_depth: OrderDepth, product: str):
            mm_bid_price, mm_bid_qty = max(order_depth.buy_orders.items(), key = lambda x: x[1])
            mm_ask_price, mm_ask_qty = max(order_depth.sell_orders.items(), key = lambda x: x[1])

            return mm_bid_price, mm_ask_price

    def executor(self, order_depth: OrderDepth, product: str, position: int, traderData: str): # add product to execute set of strategies
        orders: List[Order] = []

        POS_LIMIT = {"KELP": 50, RR: 50}

        if product == "KELP":
            logger.print("Touching -grass- kelp\n")
            mm_bot_bid, mm_bot_sell = self.identify_mm_bot(order_depth, product)

            if position == 0:
                if len(order_depth.sell_orders) > 0:
                    share_price = min(order_depth.sell_orders.keys())
                    orders.append(Order(product, share_price, 1))
                    traderData = str(share_price)

            
            fair_price = (mm_bot_bid + mm_bot_sell) / 2

            logger.print("MM Fair Price: ", fair_price)

            

        return orders, traderData
    

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        traderData = state.traderData
        logger.print("traderData: " + traderData)
        logger.print("Observations: " + str(state.observations))

        result = {}
        for product in state.order_depths:
            if product == "KELP":
                order_depth: OrderDepth = state.order_depths[product]
                position = state.position[product] if product in state.position else 0

                result[product], traderData = self.executor(order_depth, product, position, traderData)   
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
