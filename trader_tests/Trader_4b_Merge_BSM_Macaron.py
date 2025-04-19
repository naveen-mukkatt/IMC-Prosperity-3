from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string, json, math, queue, statistics
import numpy as np

# flag must be set to true before submitting
submission = False

if submission:
    # parameters necessary for submission, do NOT CHANGE
    verbose_level = 2
    log_iter = 1
else:
    # user customizable parameters
    verbose_level = 0
    log_iter = 1

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
            if verbose_level == 2: # oh
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
                if self.logs != "":
                    print (f"Time: {state.timestamp}")
                    print (self.logs)



        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            [],
            [],
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

def log(*strings, verbose=1):
    if verbose <= verbose_level:
        logger.print(*strings)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

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

    def orderbook_buy_size(self):
        return sum(self.order_depth.buy_orders.values())
    def orderbook_sell_size(self):
        return -sum(self.order_depth.sell_orders.values())
    
    def limit_buy_orders(self):
        return self.limit - self.position - self.nbuy 
    def limit_sell_orders(self):
        return self.limit + self.position - self.nsell
    
    def max_buy_orders(self):
        return min(self.limit_buy_orders(), self.orderbook_sell_size())
    def max_sell_orders(self):
        return min(self.limit_sell_orders(), self.orderbook_buy_size())
    
    def buy(self, price: int, quantity: int, print: bool=False):
        if print:
            log("Buy Order: ", price, quantity)     
        if quantity > self.limit_buy_orders():
            log("Buy Order: ", price, quantity, " exceeds max buy orders")
        elif quantity > 0 and quantity <= self.limit_buy_orders():
            self.orders.append(Order(self.product, int(price), quantity))
            self.nbuy += quantity
    def sell(self, price: int, quantity: int, print: bool=False):
        if print:
            log("Sell Order: ", price, quantity)
        if quantity > self.limit_sell_orders():
            log("Sell Order: ", price, quantity, " exceeds max sell orders")
        elif quantity > 0 and quantity <= self.limit_sell_orders():
            self.orders.append(Order(self.product, int(price), -quantity))
            self.nsell += quantity
    
    def full_buy(self, quantity: int):
        """Buy the orderbook until the quantity of shares are bought. Limited by max_buy_orders."""
        q = quantity
        for price, volume in self.order_depth.sell_orders.items():
            if volume < 0:
                buy_vol = min(q, min(self.limit_buy_orders(), -volume))
                self.buy(price, buy_vol)
                q -= buy_vol
                if q <= 0 or self.limit_buy_orders() <= 0:
                    break
    def full_sell(self, quantity: int):
        """Sell the orderbook until the quantity of shares are sold. Limited by max_sell_orders."""
        q = quantity
        for price, volume in self.order_depth.buy_orders.items():
            if volume > 0:
                sell_vol = min(q, min(self.limit_sell_orders(), volume))
                self.sell(price, sell_vol)
                q -= sell_vol
                if q <= 0 or self.limit_sell_orders() <= 0:
                    break

    def cancel_orders(self):
        """Cancel all orders."""
        self.orders = []
        self.nsell = 0
        self.nbuy = 0   
    def cancel_buy_orders(self):
        """Cancel all buy orders."""
        self.orders = [order for order in self.orders if order.quantity < 0]
        self.nsell = 0
        self.nbuy = 0
    def cancel_sell_orders(self):
        """Cancel all sell orders."""
        self.orders = [order for order in self.orders if order.quantity > 0]
        self.nsell = 0
        self.nbuy = 0

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

        bid_val = fair_val - edge
        ask_val = fair_val + edge

        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > bid_val or (bid_price == bid_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == bid_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    self.sell(bid_price, sell_vol)

        for ask_price, ask_vol in self.order_depth.sell_orders.items():
            ask_vol *= -1
            if ask_price < ask_val or (ask_price == ask_val and self.active_position() < 0):
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if ask_price == ask_val:
                    buy_vol = min(buy_vol, -self.active_position())
                if buy_vol > 0:
                    self.buy(ask_price, buy_vol)

    def market_make(
        self,
        buy_price: int,
        sell_price: int
    ):
        self.buy(buy_price, self.limit_buy_orders())
        self.sell(sell_price, self.limit_sell_orders())
    
    def market_make_undercut(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1

        return self.market_make(mm_buy, mm_sell)
    
    def fair_val(self):
        if len(self.order_depth.buy_orders) == 0 or len(self.order_depth.sell_orders) == 0:
            return self.hist_mid[-1] if len(self.hist_mid) > 0 else math.nan
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
        self.hist_mm_mid.append(self.fair_val())
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
                value = self.fair_val()
            else:
                value = self.mid_price()
        offset = max(1, (self.best_ask() - self.best_bid()) // 2)
        buy_price = round(value - offset)
        sell_price = round(value + offset)
        order_size = min(self.limit_buy_orders(), self.limit_sell_orders())
        self.buy(buy_price, order_size)
        self.sell(sell_price, order_size)

    def buy_one(self):
        """Utility function to buy one share at the start and test PnL."""
        if self.position == 0:
            if len(self.order_depth.sell_orders) > 0:
                self.buy(self.best_ask(), 1)

    def sell_one(self):
        if self.position == 0:
            if len(self.order_depth.buy_orders) > 0:
                self.sell(self.best_bid(), 1)

    def strategy(self): # DO NOT REMOVE ERROR HANDLING
        raise NotImplementedError("Strategy not implemented")

    def execute(self, blank: bool=False): 
        if blank:
            return [], ""
        self.strategy()
    
    def getData(self):
        return self.return_mids()

class Resin(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)

class Kelp(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)

class MeanReversion(Product): # defunct for squid ink
    def __init__(self, symbol: str, limit: int, state: TradingState, gamma: float, window: int):
        super().__init__(symbol, limit, state)
        self.gamma = gamma
        self.window = window

    def avg_midprice(self):
        return sum(self.hist_mid)/len(self.hist_mid) if len(self.hist_mid) > 0 else self.mid_price()
    
    def ou(self):
        z = self.compute_z_score() # idk how to use this yet, bid sizing maybe

        # OU adjustment: reversion toward fair value
        adj_price = self.mid_price() + self.gamma * (self.avg_midprice() - self.mid_price())

        spread = 1 
        bid_quote = int(adj_price - spread)
        ask_quote = int(adj_price + spread)
        bid_vol = min(20, self.limit_buy_orders())
        ask_vol = min(20, self.limit_sell_orders())

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
        ...

class Jam(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        ...

class Djembe(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        ...
class Basket1(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        ...
        
class Basket2(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def strategy(self):
        ...

class ArbStrategy():
    def __init__(self, strat: str, arb_prods: List[Product], arb_coefs: List[float], mean: float, std: float, cutoffs: tuple):
        self.strat = strat
        self.arb_prods = arb_prods
        self.arb_coefs = arb_coefs
        self.mean = mean
        self.std = std
        self.cutoffs = cutoffs
    
    def z_score(self):
        total = 0
        for prod, coef in zip(self.arb_prods, self.arb_coefs):
            total += coef * prod.mid_price()
        z = (total - self.mean) / self.std
        return z
    
    def buy_val(self):
        # Calculate value of long position
        total = 0
        for prod, coef in zip(self.arb_prods, self.arb_coefs):
            if coef > 0:
                total += coef * prod.best_ask()
            else:
                total += coef * prod.best_bid()
        return total
    def sell_val(self):
        # Calculate value of short position
        total = 0
        for prod, coef in zip(self.arb_prods, self.arb_coefs):
            if coef > 0:
                total += coef * prod.best_bid()
            else:
                total += coef * prod.best_ask()
        return total
    
    def signal(self):
        z = self.z_score()
        if abs(z) <= self.cutoffs[0]:
            return 0 # zero position
        elif abs(z) > self.cutoffs[1]:
            if z > 0: 
                return 1 # sell
            else:
                return -1 # buy
        else:
            return -99 # no signal
    
    def arbitrage(self):
        log("Arbitrage Signal and Z-Score: ", self.strat, self.signal(), self.z_score(), verbose=2)
        sgn = self.signal()
        if sgn == 1:
            # find max short quantity that's safe without overflowing
            max_short = min([
                prod.max_sell_orders() // coef if coef > 0
                else prod.max_buy_orders() // abs(coef)
                for prod, coef in zip(self.arb_prods, self.arb_coefs)
                if coef != 0
            ])            
            
            for prod, coef in zip(self.arb_prods, self.arb_coefs):
                if coef > 0:
                    prod.full_sell(int(coef * max_short))
                elif coef < 0:
                    prod.full_buy(int(-coef * max_short))
        
        elif sgn == -1:
            # find max long quantity that's safe without overflowing
            max_long = min([
                prod.max_buy_orders() // coef if coef > 0 
                else prod.max_sell_orders() // abs(coef)
                for prod, coef in zip(self.arb_prods, self.arb_coefs)
                if coef != 0
            ])

            for prod, coef in zip(self.arb_prods, self.arb_coefs):
                if coef > 0:
                    prod.full_buy(int(coef * max_long))
                elif coef < 0:
                    prod.full_sell(int(-coef * max_long))

        elif sgn == 0:
            base = self.arb_prods[-1].active_position() # negative if need to sell, positive if need to buy
            log(base, verbose=3)

            if abs(base) < 10:
                return
            
            max_pos = 10000
            for prod, coef in zip(self.arb_prods, self.arb_coefs):
                if coef * base > 0:
                    test_prod = prod.max_buy_orders() // abs(coef)
                    if test_prod < max_pos:
                        max_pos = test_prod
                elif coef * base < 0:
                    test_prod = prod.max_sell_orders() // abs(coef)

                    if test_prod < max_pos:
                        max_pos = test_prod

            # max_pos = min([
            #     prod.max_buy_orders() // abs(coef) if coef * base > 0
            #     else prod.max_sell_orders() // abs(coef)
            #     for prod, coef in zip(self.arb_prods, self.arb_coefs)
            # ])

            log("Zeroing Position: ", max_pos, verbose=2)

            for prod, coef in zip(self.arb_prods, self.arb_coefs):
                if coef * base > 0:
                    prod.full_buy(int(abs(coef) * max_pos))
                elif coef * base < 0:
                    prod.full_sell(int(abs(coef) * max_pos))

class BasketArb():
    def __init__(self, symbol: str, croissant: Product, jam: Product, djembe: Product, basket1: Product, basket2: Product):
        self.symbol = symbol
        self.croissant = croissant
        self.jam = jam
        self.djembe = djembe
        self.basket1 = basket1
        self.basket2 = basket2

    def execute_arb(self):
        strats = ["ARB1", "ARB2"]
        arb_prods = {
            "ARB1": [self.croissant, self.jam, self.djembe, self.basket1],
            "ARB2": [self.croissant, self.jam, self.basket2],
        }
        arb_coefs = {
            "ARB1": [6, 3, 1, -1],
            "ARB2": [4, 2, -1],
        }
        mean = {
            "ARB1": -48,
            "ARB2": -30,
        }
        std = {
            "ARB1": 85,
            "ARB2": 60,
        }
        cutoffs = {
            "ARB1": (0.25, 1.75),
            "ARB2": (0.25, 1.75)
        }
        arb_strats = {
            strat: ArbStrategy(strat, arb_prods[strat], arb_coefs[strat], mean[strat], std[strat], cutoffs[strat])
            for strat in strats
        }
        # Execute both
        for strat in ["ARB1", "ARB2"]:
            arb_strats[strat].arbitrage()
        
        
    def execute(self):
        self.execute_arb()
    
    def getData(self):
        return "Arb done"
  
class Option(Product):
    def __init__(self, symbol: str, limit: str, strike: float, underlying: Product, state: TradingState, a: float, b: float, c: float):
        super().__init__(symbol, limit, state)
        self.strike = strike
        self.underlying = underlying
        self.state = state
        self.window = 100
        self.a = a
        self.b = b
        self.c = c

    def TTE(self):
        """Returns time to expiration in days (365 trading days, per @debnandy)."""
        return 6 - (self.state.timestamp / 1000000)  
    
    def BSM(self, S, K, T, sigma):
        """Black-Scholes on European calls."""
        d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm_cdf(d1) - K * norm_cdf(d2)
    
    def iv(self, S, K, T, price, max_iter=1000):
        """Inverts BSM to find IV given every other parameter."""
        low = 1e-6
        high = 5.0 
        log("Price fed into IV algorithm = ", price, verbose=2)

        for i in range(max_iter):
            sigma = (low + high) / 2
            price_est = self.BSM(S, K, T, sigma)
            if abs(price - price_est) < 1e-6:
                return sigma
            if price_est < price:
                low = sigma
            else:
                high = sigma
        return -999999999999999 # error

    def moneyness(self, S, K, T):
        """Calculates moneyness of an option."""
        return np.log(K / S) / np.sqrt(T)

    def model_iv(self):
        m = self.moneyness(self.strike, self.underlying.mid_price(), self.TTE())
        return self.a * m**2 + self.b * m + self.c

    def iv_diff(self):
        if self.underlying.mid_price() == math.nan:
            log("MIDPRICE = NAN", verbose=0)
            return 0.0001
        v = self.iv(self.underlying.mid_price(), self.strike, self.TTE(), self.mid_price())
        if v < -100000:
            log("midprice = nan??", verbose=0)
            return 0.0001
        return v - self.model_iv()
    
    def fair_val(self):
        """Implements above BSM with appropriate parameters."""
        S = self.underlying.mid_price()
        K = self.strike
        T = self.TTE()
        sigma = self.model_iv()
        return self.BSM(S, K, T, sigma)
    
    def act(self):
        ivd = self.iv_diff()
        fv = self.fair_val()

        log(f"IV Diff: {ivd:.4f}, Fair Value: {fv}, Mid Price: {self.mid_price()}, Best Bid/Ask: {self.best_bid()}/{self.best_ask()}", verbose=0)
        if ivd > 0.0002: # sell signal. Check best_bid.
            if self.best_bid() > fv:
                self.sell(self.best_bid(), self.max_sell_orders())
            else:
                self.sell(fv, self.max_sell_orders())
        elif ivd < -0.0002: # buy signal. Check best_ask.
            if self.best_ask() < fv:
                self.buy(self.best_ask(), self.max_buy_orders())
            else:
                self.buy(fv, self.max_buy_orders())
        else: 
            # neutralize
            if abs(ivd) < 0.00005:
                if self.position > 0:
                    self.sell(fv, self.max_sell_orders())
                elif self.position < 0:
                    self.buy(fv, self.max_buy_orders())
    def strategy(self):
        ...
    
    def execute(self):
        ...  
        
class Rock(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
    
    def strategy(self):
        ...

class BlackScholes():
    def __init__(self, symbol: str, underlying: Rock, options: List[Option]):
        self.symbol = symbol
        self.options = options
        self.underlying = underlying
        self.state = underlying.state
        self.window = 100

    def execute(self):
        for option in self.options:
            option.act()

class Macaron(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)
        
    def strategy(self):
        fvx = self.fair_val()
        self.market_take(fvx)
        self.market_make_undercut(fvx, 1)
    
    def obtain_position_change(self):
        # scalars, one for each of a leftover const, export, import, sugar, and sun (must be hardcoded)
        bid_consts = [394.96690819, -54.1238246, -49.90954794, 3.80966583, -2.46172691]
        ask_consts = [396.50798592, -54.12486522, -49.91003338, 3.80961465, -2.46185596]

        sanityCheck = self.state.observations.conversionObservations
        if 'MAGNIFICENT_MACARONS' not in sanityCheck:
            raise NotImplementedError("Macarons not found")

        # obs: bidPrice, askPrice, transportFees, exportTariff, importTariff, sugarPrice, sunlightIndex
        obs = sanityCheck['MAGNIFICENT_MACARONS']
        rel_vals = [obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        
        # get a fair bid/ask, then take average
        fair_bid = bid_consts[0]
        for i in range(1, 5):
            fair_bid += (rel_vals[i - 1] * bid_consts[i])
        fair_ask = ask_consts[0]
        for i in range(1, 5):
            fair_ask += (rel_vals[i - 1] * ask_consts[i])
        
        # compute when it's good to buy / sell stuff
        effective_bid = obs.askPrice + obs.transportFees + obs.importTariff
        effective_ask = obs.bidPrice - obs.transportFees - obs.exportTariff

        # there might (?) be scalars for differences between fair_val and effective_bid/ask
        pos_change = 0
        # good to buy, go long
        if fair_bid > effective_bid:
            pos_change = min((fair_bid - effective_bid) / 15, 10)
        # good to sell, go short
        if fair_ask < effective_ask:
            pos_change = max((fair_ask - effective_ask) / 15, -10)

        log(f"pineapple", pos_change, verbose=0)
        return max(-10, min(10, int(round(pos_change))))

def create_products(state: TradingState):
    products = {}
    products["RAINFOREST_RESIN"] = Resin("RAINFOREST_RESIN", 50, state)
    products["KELP"] = Kelp("KELP", 50, state)
    products["SQUID_INK"] = Ink("SQUID_INK", 50, state)
    products["CROISSANTS"] = Croissant("CROISSANTS", 250, state)
    products["JAMS"] = Jam("JAMS", 350, state)
    products["DJEMBES"] = Djembe("DJEMBES", 60, state)
    products["PICNIC_BASKET1"] = Basket1("PICNIC_BASKET1", 60, state)
    products["PICNIC_BASKET2"] = Basket2("PICNIC_BASKET2", 100, state)
    products["BASKET_ARB"] = BasketArb("BASKET_ARB", 
                                        products["CROISSANTS"],
                                        products["JAMS"],
                                        products["DJEMBES"],
                                        products["PICNIC_BASKET1"],
                                        products["PICNIC_BASKET2"])
    products["VOLCANIC_ROCK"] = Rock("VOLCANIC_ROCK", 400, state)
    strikes = [9500, 9750, 10000, 10250, 10500]
    for strike in strikes:
        products["VOLCANIC_ROCK_VOUCHER_" + str(strike)] = Option("VOLCANIC_ROCK_VOUCHER_" + str(strike), 
                                                                    200, 
                                                                    strike, 
                                                                    products["VOLCANIC_ROCK"], 
                                                                    state, 
                                                                    0.0001, 0.0001, 0.0001)
    products["BSM"] = BlackScholes("BSM", 
                                    products["VOLCANIC_ROCK"],
                                    [products["VOLCANIC_ROCK_VOUCHER_" + str(strike)] for strike in strikes])
    products["MAGNIFICENT_MACARONS"] = Macaron("MAGNIFICENT_MACARONS", 50, state)

    return products

class Trader:
        
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #log("traderData: " + state.traderData, 2)
        #log("Observations: " + str(state.observations), 2)

        result = {}

        traderData = ""
        product_instances = create_products(state)

        for product, instance in product_instances.items():
            if product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "BASKET_ARB", "BSM", "MAGNIFICENT_MACARONS"]:
                instance.execute()

        for product, instance in product_instances.items():   
            # check if instance is instance of Product
            if isinstance(instance, Product):
                traderData += instance.getData() + "\n"
                result[product] = instance.orders

        # NOW we figure out conversion stuff
        conversions = product_instances["MAGNIFICENT_MACARONS"].obtain_position_change()
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData