from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import json
import math
import queue
import numpy as np
from abc import ABC, abstractmethod

# flag must be set to true before submitting
submission = True

if submission == True:
    # parameters necessary for submission, do NOT CHANGE
    verbose_level = 2
    log_iter = 1
else:
    # user customizable parameters
    verbose_level = 0
    log_iter = 10000

# verbosity level:
# 0 - zero output by default (can use for debugging)
# 1 - log normal operations
# 2 - print EVERYTHING required for prosperity imc
# log on timestamps that are multiples of log_iter


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

        if (state.timestamp % log_iter) == 0:
            if verbose_level == 2:
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
            else:
                print (f"Time: {state.timestamp}")
                print (self.logs)


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

def log(string, verbose=1):
    if verbose <= verbose_level:
        logger.print(string)

class Product():
    def __init__(self, product: str, limit: int, state: TradingState):
        self.state = state
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
        self.hist_sum = 0
        self.hist_sum_squared = 0
        self.hist_mean = 0
        self.hist_std = 0
        self.init_hist_mid()
        self.orders: List[Order] = []
        self.window = 10
    
    def buy(self, price: int, quantity: int):
        if quantity <= self.max_buy_orders():
            self.orders.append(Order(self.product, int(price), quantity))
            self.nbuy += quantity
    def sell(self, price: int, quantity: int):
        if quantity <= self.max_sell_orders():
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
        
    def market_take(self, fair_val: float, edge: float = 0):
        log("Market Taking:", 1)

        bid_val = fair_val - edge
        ask_val = fair_val + edge

        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > bid_val or (bid_price == bid_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == bid_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    log(f"Market take: selling {sell_vol}@{bid_price}\n", 1)
                    self.sell(bid_price, sell_vol)


        for ask_price, ask_vol in self.order_depth.sell_orders.items():
            ask_vol *= -1
            if ask_price < ask_val or (ask_price == ask_val and self.active_position() < 0):
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if ask_price == ask_val:
                    buy_vol = min(buy_vol, -self.active_position())
                if buy_vol > 0:
                    log(f"Market take: buying {buy_vol}@{ask_price}\n", 1)
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
        edge: float = 0,
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
            for price in prods[ind].split(" ")[3:]:
                self.hist_mid.append(float(price))
            for mm_price in prods[ind + 1].split(" ")[3:]:
                self.hist_mm_mid.append(float(mm_price))

            iter = self.state.timestamp / 100

            self.hist_sum = float(prods[ind].split(" ")[1])
            self.hist_sum_squared = float(prods[ind].split(" ")[2])
            self.hist_mean = self.hist_sum / iter
            self.hist_std = math.sqrt(self.hist_sum_squared / iter - self.hist_mean ** 2)

    def return_mids(self):
        """Converts self.hist_mid and self.hist_mm_mid into string form for TraderData.
        String form: [PRODUCT] [sum] [sum^2] [mp1] [mp2] ... [mp10]
        [PRODUCT] [sum] [sum^2] [mm_mp1] [mm_mp2] ... [mm_mp10]"""
        self.hist_mid.append(self.mid_price())
        self.hist_mm_mid.append(self.max_vol_mid())
        self.hist_mid = self.hist_mid[-self.window:]
        self.hist_mm_mid = self.hist_mm_mid[-self.window:]
        res = self.product
        res += " " + str(self.mid_price() + self.hist_sum)
        res += " " + str(self.mid_price() ** 2 + self.hist_sum_squared)
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

    def buy_one(self):
        if self.position == 0:
            if len(self.order_depth.sell_orders) > 0:
                self.buy(self.best_ask(), 1)

    def sell_one(self):
        if self.position == 0:
            if len(self.order_depth.buy_orders) > 0:
                self.sell(self.best_bid(), 1)

    def fair_val(self): # Children inherit the default fair_val   
        mid = self.max_vol_mid()
        prev_mid = mid
        if len(self.hist_mm_mid) > 0:
            prev_mid = self.hist_mm_mid[-1]

        val = mid * 0.9 + prev_mid * 0.1
        return val

    def strategy(self): # RUNTIME POLYMORPHISM BTW
        raise NotImplementedError()
        ...

    def execute(self, blank: bool=False): 
        if blank:
            return [], ""
        self.strategy()
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
        #self.buy_one()

class MeanReversion(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState, gamma: float, window: int):
        super().__init__(symbol, limit, state)
        self.gamma = gamma
        self.window = window

    def avg_midprice(self):
        return sum(self.hist_mid)/len(self.hist_mid) if len(self.hist_mid) > 0 else self.mid_price()
    
    def ou(self):

        z = self.compute_z_score()

        # OU adjustment: reversion toward fair value
        adj_price = self.mid_price() + self.gamma * (self.avg_midprice() - self.mid_price())

        # Create bid/ask around adjusted price
        spread = 1  # you can tune this
        bid_quote = int(adj_price - spread)
        ask_quote = int(adj_price + spread)
        bid_vol = min(20, self.max_buy_orders())
        ask_vol = min(20, self.max_sell_orders())

        self.buy(bid_quote, bid_vol)
        self.sell(ask_quote, ask_vol)

    def compute_z_score(self):
        if len(self.hist_mid) < self.window:
            return 0
        recent = np.array(self.hist_mid)
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0
        return (self.mid_price() - mean) / std

class BuyLowSellHigh(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def buy_low_sell_high(self, mean_diff : float = -1):
        if len(self.hist_mid) < self.window:
            return
        if mean_diff < 0:
            mean_diff = self.hist_std

        lo = self.hist_mean - mean_diff
        hi = self.hist_mean + mean_diff

        if self.best_ask() < lo:
            self.buy(self.best_ask(), self.max_buy_orders())
        elif self.best_bid() > hi:
            self.sell(self.best_bid(), self.max_sell_orders())

class Ink(BuyLowSellHigh):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
        self.window = 10
    
    def fair_val(self):      
        return -1  
    
    def strategy(self):
        self.buy_low_sell_high(mean_diff = 10)

class Croissant(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        fv1 = arbitrage(["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"], [6, 3, 1, -1], self.state)
        if fv1 < -50:
            for i in range(6):
                self.buy_one()
        if fv1 > 50:
            for i in range(6):
                self.sell_one()
        
        # going to add more here

        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)

class Jam(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        fv = arbitrage(["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"], [6, 3, 1, -1], self.state)
        if fv < -50:
            for i in range(3):
                self.buy_one()
        if fv > 50:
            for i in range(3):
                self.sell_one()

class Djembe(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        fv = arbitrage(["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"], [6, 3, 1, -1], self.state)
        if fv < -50:
            for i in range(1):
                self.buy_one()
        if fv > 50:
            for i in range(1):
                self.sell_one()
        
        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)

class Basket1(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        fv = arbitrage(["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"], [6, 3, 1, -1], self.state)
        if fv < -50:
            for i in range(1):
                self.sell_one()
        if fv > 50:
            for i in range(1):
                self.buy_one()
        
        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)
        
class Basket2(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        ...
        #self.buy_one()

class BasketArb(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState, croissant: Product, jam: Product, djembe: Product, basket1: Product, basket2: Product):
        super().__init__(symbol, limit, state)
        self.croissant = croissant
        self.jam = jam
        self.djembe = djembe
        self.basket1 = basket1
        self.basket2 = basket2

    # zero_relation: List - contains coefficients on each item, should result in zero of all items at the end

    def arbitrage(product_list: List[str], coefs: List[float]):
        
        ...

    def choose_arb(self):
        
        return self.choose_arb()
    
    def gen_signal(self, product_list: List[str], coefs: List[float]):
        """returns -1 """
    def strategy(self):
        fv = arbitrage(["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"], [6, 3, 1, -1], self.state)
        if fv < -50:
            for i in range(1):
                self.buy_one()
        if fv > 50:
            for i in range(1):
                self.sell_one()
        
        fvx = self.fair_val()


def create_products(state: TradingState):
    products = {}
    for product, (cls, limit) in round_1_product_classes.items():
        products[product] = cls(product, limit, state)
    return products

round_1_product_classes = {
    "RAINFOREST_RESIN": (Resin, 50),
    "KELP": (Kelp, 50),
    "SQUID_INK": (Ink, 50),
}

round_2_product_classes = {
    "RAINFOREST_RESIN": (Resin, 50),
    "KELP": (Kelp, 50),
    "SQUID_INK": (Ink, 50),
    "CROISSANTS": (Croissant, 250),
    "JAMS": (Jam, 350),
    "DJEMBES": (Djembe, 60),
    "PICNIC_BASKET1": (Basket1, 60),
    "PICNIC_BASKET2": (Basket2, 100)
}

class Trader:
        
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        log("traderData: " + state.traderData, 2)
        log("Observations: " + str(state.observations), 2)

        result = {}

        traderData = ""
        product_instances = create_products(state)
        for product, instance in product_instances.items():
            orders, data_prod = instance.execute(blank=False)
            traderData += data_prod + "\n"
            result[product] = orders   
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
