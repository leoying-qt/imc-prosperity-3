#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

from datamodel import OrderDepth, TradingState, Order
from typing import List, Tuple
import jsonpickle
import math
import numpy as np

# Global constant for the common rolling lookback window (set to 100 ticks)
OBSERVATION_WINDOW_COMMON = 100

# ------------------------- Helper Functions -------------------------

def get_mid_price(order_depth: OrderDepth) -> float:
    """
    Compute the mid-price from the order depth if both buy and sell orders are available.
    """
    if order_depth.buy_orders and order_depth.sell_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    return None

def norm_cdf(x: float) -> float:
    """Standard Normal cumulative distribution using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """
    Compute the Black–Scholes call price with risk-free rate r = 0.
    For T <= 0 or sigma <= 0, returns max(0, S - K).
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def implied_volatility_bisection(S: float, market_price: float, K: float, T: float,
                                 tol: float = 0.001, max_iter: int = 100) -> float:
    """
    Calculate the implied volatility by the bisection method, to within an absolute tolerance of tol.
    """
    low = 0.001
    high = 5.0
    mid = (low + high) / 2
    for i in range(max_iter):
        mid = (low + high) / 2
        price = bs_call_price(S, K, T, mid)
        if (high - low) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return mid

def compute_tte(day: int, timestamp: int) -> float:
    """
    Compute time to expiry (TTE) based on day and timestamp.
    For example, day 0, time 0: TTE = 8/250; day 0, time 999000: TTE = 7/250; etc.
    """
    base_tte = (8 - day) / 250.0
    decay = (timestamp / 999000.0) * (1 / 250.0)
    return base_tte - decay

# ------------------------- Main Trading Classes -------------------------

class Trader:
    BASKET1_SYMBOL = "PICNIC_BASKET1"
    BASKET2_SYMBOL = "PICNIC_BASKET2"

    CROISSANT_SYMBOL = "CROISSANTS"
    DJEMBE_SYMBOL = "DJEMBES"
    JAM_SYMBOL = "JAMS"

    VOLCANIC_ROCK_SYMBOL = "VOLCANIC_ROCK"
    # Vouchers we actively trade (for vega mean-reversion)
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    # Delta-one hedging instruments (assumed delta-one)
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"

    def __init__(self):
        self.round1_trader = Round1Trader()

        self.basket1_trader = BasketTrader(
            basket_symbol=self.BASKET1_SYMBOL,
            synthetic_weights={self.CROISSANT_SYMBOL: 6, self.JAM_SYMBOL: 3, self.DJEMBE_SYMBOL: 1},
            baseline_mean=0,
            window=50,
            zscore_threshold=1.0,
            target_position=60
        )

        self.basket2_trader = BasketTrader(
            basket_symbol=self.BASKET2_SYMBOL,
            synthetic_weights={self.CROISSANT_SYMBOL: 4, self.JAM_SYMBOL: 2},
            baseline_mean=0,
            window=50,
            zscore_threshold=1.0,
            target_position=100
        )

    def run(self, state: TradingState):
        result = {}

        resin_orders = self.round1_trader.rainforest_resin_strat(state)
        result[self.round1_trader.RESIN_SYMBOL] = resin_orders

        kelp_orders = self.round1_trader.kelp_strat(state)
        result[self.round1_trader.KELP_SYMBOL] = kelp_orders

        if self.round1_trader.SQUID_INK_SYMBOL in state.order_depths:
            squid_orders, squid_traderData = self.round1_trader.squid_ink_strat(state)
            result[self.round1_trader.SQUID_INK_SYMBOL] = squid_orders
            trader_data = {}
            if state.traderData and state.traderData != "":
                trader_data = jsonpickle.decode(state.traderData)
            trader_data[self.round1_trader.SQUID_INK_SYMBOL] = jsonpickle.decode(squid_traderData)[self.round1_trader.SQUID_INK_SYMBOL]
            state.traderData = jsonpickle.encode(trader_data)

        self.basket1_trader.update(state, result)
        self.basket2_trader.update(state, result)

        # New VOLCANIC_ROCK price-based mean-reversion strategy
        vr_orders, vr_data = self.round1_trader.volcanic_rock_strat(state)
        result[self.VOLCANIC_ROCK_SYMBOL] = vr_orders

        # New voucher vega mean-reversion strategies for strikes 10000, 10250, 10500.
        v10000_orders, v10000_data, delta_10000 = self.round1_trader.voucher_vega_strat(state, self.VOUCHER_10000, 10000)
        result[self.VOUCHER_10000] = v10000_orders

        v10250_orders, v10250_data, delta_10250 = self.round1_trader.voucher_vega_strat(state, self.VOUCHER_10250, 10250)
        result[self.VOUCHER_10250] = v10250_orders

        v10500_orders, v10500_data, delta_10500 = self.round1_trader.voucher_vega_strat(state, self.VOUCHER_10500, 10500)
        result[self.VOUCHER_10500] = v10500_orders

        # Sum net delta exposure from voucher trades.
        net_delta = delta_10000 + delta_10250 + delta_10500

        # Delta hedge every 10 time steps.
        if state.timestamp % 10 == 0:
            hedge_orders = self.round1_trader.hedge_delta(state, net_delta,
                                                            self.VOLCANIC_ROCK_SYMBOL,
                                                            self.VOUCHER_9500,
                                                            self.VOUCHER_9750)
            if hedge_orders:
                if self.VOLCANIC_ROCK_SYMBOL in result:
                    result[self.VOLCANIC_ROCK_SYMBOL].extend(hedge_orders)
                else:
                    result[self.VOLCANIC_ROCK_SYMBOL] = hedge_orders

        # Merge updated traderData from new strategies.
        combined_data = {}
        if state.traderData and state.traderData != "":
            combined_data = jsonpickle.decode(state.traderData)
        combined_data[self.VOLCANIC_ROCK_SYMBOL] = jsonpickle.decode(vr_data)[self.VOLCANIC_ROCK_SYMBOL]
        combined_data[self.VOUCHER_10000] = jsonpickle.decode(v10000_data)[self.VOUCHER_10000]
        combined_data[self.VOUCHER_10250] = jsonpickle.decode(v10250_data)[self.VOUCHER_10250]
        combined_data[self.VOUCHER_10500] = jsonpickle.decode(v10500_data)[self.VOUCHER_10500]
        state.traderData = jsonpickle.encode(combined_data)

        conversions = 1
        traderData = state.traderData if state.traderData else ""
        return result, conversions, traderData

class Round1Trader:
    """
    Implements:
      1. RAINFOREST_RESIN Strategy.
      2. KELP Strategy.
      3. SQUID_INK Strategy.
      4. VOLCANIC_ROCK Price-Based Mean-Reversion Strategy.
      5. Voucher Vega Mean-Reversion Strategy (for strikes 10000, 10250, 10500).

    The new strategies use a common rolling lookback window of length OBSERVATION_WINDOW_COMMON.
    """
    # ---- RAINFOREST_RESIN parameters ----
    RESIN_SYMBOL = "RAINFOREST_RESIN"
    RESIN_FAIR_PRICE = 10000
    RESIN_TAKE_WIDTH = 1
    RESIN_CLEAR_WIDTH = 0
    RESIN_DISREGARD_EDGE = 1
    RESIN_JOIN_EDGE = 2
    RESIN_DEFAULT_EDGE = 4
    RESIN_SOFT_POSITION_LIMIT = 20
    RESIN_POSITION_LIMIT = 50

    # ---- KELP parameters ----
    KELP_SYMBOL = "KELP"
    KELP_TAKE_WIDTH = 1
    KELP_CLEAR_WIDTH = 0
    KELP_PREVENT_ADVERSE = True
    KELP_ADVERSE_VOLUME = 15
    KELP_REVERSION_BETA = -0.25
    KELP_DISREGARD_EDGE = 1
    KELP_JOIN_EDGE = 0
    KELP_DEFAULT_EDGE = 1
    KELP_POSITION_LIMIT = 50

    # ---- SQUID_INK parameters ----
    SQUID_INK_SYMBOL = "SQUID_INK"
    SQUID_INK_TAKE_WIDTH = 1
    SQUID_INK_CLEAR_WIDTH = 0
    SQUID_INK_OBSERVATION_WINDOW = 750
    SQUID_INK_REVERSION_THRESHOLD = 2
    SQUID_INK_POSITION_LIMIT = 50

    def __init__(self):
        self.kelp_last_price = None
        self.LIMIT = {
            self.RESIN_SYMBOL: self.RESIN_POSITION_LIMIT,
            self.KELP_SYMBOL: self.KELP_POSITION_LIMIT,
            self.SQUID_INK_SYMBOL: self.SQUID_INK_POSITION_LIMIT,
        }
        self.VOLCANIC_ROCK_POSITION_LIMIT = 400
        self.VOUCHER_POSITION_LIMIT = 200
        # New parameter: voucher_scaler scales the order quantity.
        self.voucher_scaler = 5  # Adjust this as needed; for example, set to 10 for 10x scaling.
        # New parameter: voucher_zscore_threshold now customizable (default 2)
        self.voucher_zscore_threshold = 1

    # ---------------------------
    # RAINFOREST_RESIN Strategy Methods
    # ---------------------------
    def take_orders_resin(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            available_sell = -order_depth.sell_orders[best_ask]
            if best_ask <= self.RESIN_FAIR_PRICE - self.RESIN_TAKE_WIDTH:
                max_buy = self.RESIN_POSITION_LIMIT - position
                quantity = min(available_sell, max_buy)
                if quantity > 0:
                    orders.append(Order(self.RESIN_SYMBOL, int(round(best_ask)), quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            available_buy = order_depth.buy_orders[best_bid]
            if best_bid >= self.RESIN_FAIR_PRICE + self.RESIN_TAKE_WIDTH:
                max_sell = self.RESIN_POSITION_LIMIT + position
                quantity = min(available_buy, max_sell)
                if quantity > 0:
                    orders.append(Order(self.RESIN_SYMBOL, int(round(best_bid)), -quantity))
                    sell_order_volume += quantity
        return orders, buy_order_volume, sell_order_volume

    def clear_orders_resin(self, order_depth: OrderDepth, position: int,
                           buy_order_volume: int, sell_order_volume: int):
        orders: List[Order] = []
        net_position = position + buy_order_volume - sell_order_volume
        if net_position > 0:
            clearing_quantity = sum(volume for price, volume in order_depth.buy_orders.items()
                                    if price >= self.RESIN_FAIR_PRICE + self.RESIN_CLEAR_WIDTH)
            clearing_quantity = min(clearing_quantity, net_position)
            max_sell = self.RESIN_POSITION_LIMIT + position
            sell_quantity = min(max_sell, clearing_quantity)
            if sell_quantity > 0:
                orders.append(Order(self.RESIN_SYMBOL,
                                    int(round(self.RESIN_FAIR_PRICE + self.RESIN_CLEAR_WIDTH)),
                                    -sell_quantity))
                sell_order_volume += sell_quantity
        if net_position < 0:
            clearing_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items()
                                    if price <= self.RESIN_FAIR_PRICE - self.RESIN_CLEAR_WIDTH)
            clearing_quantity = min(clearing_quantity, abs(net_position))
            max_buy = self.RESIN_POSITION_LIMIT - position
            buy_quantity = min(max_buy, clearing_quantity)
            if buy_quantity > 0:
                orders.append(Order(self.RESIN_SYMBOL,
                                    int(round(self.RESIN_FAIR_PRICE - self.RESIN_CLEAR_WIDTH)),
                                    buy_quantity))
                buy_order_volume += buy_quantity
        return orders, buy_order_volume, sell_order_volume

    def make_orders_resin(self, order_depth: OrderDepth, position: int,
                          buy_order_volume: int, sell_order_volume: int):
        orders: List[Order] = []
        sell_candidates = [price for price in order_depth.sell_orders.keys()
                           if price > self.RESIN_FAIR_PRICE + self.RESIN_DISREGARD_EDGE]
        buy_candidates = [price for price in order_depth.buy_orders.keys()
                          if price < self.RESIN_FAIR_PRICE - self.RESIN_DISREGARD_EDGE]
        if sell_candidates:
            best_sell = min(sell_candidates)
            sell_price = best_sell if best_sell - self.RESIN_FAIR_PRICE <= self.RESIN_JOIN_EDGE else best_sell - 1
        else:
            sell_price = self.RESIN_FAIR_PRICE + self.RESIN_DEFAULT_EDGE
        if buy_candidates:
            best_buy = max(buy_candidates)
            buy_price = best_buy if self.RESIN_FAIR_PRICE - best_buy <= self.RESIN_JOIN_EDGE else best_buy + 1
        else:
            buy_price = self.RESIN_FAIR_PRICE - self.RESIN_DEFAULT_EDGE
        if position > self.RESIN_SOFT_POSITION_LIMIT:
            sell_price -= 1
        elif position < -self.RESIN_SOFT_POSITION_LIMIT:
            buy_price += 1
        max_buy_quantity = self.RESIN_POSITION_LIMIT - (position + buy_order_volume)
        if max_buy_quantity > 0:
            orders.append(Order(self.RESIN_SYMBOL, int(round(buy_price)), max_buy_quantity))
            buy_order_volume += max_buy_quantity
        max_sell_quantity = self.RESIN_POSITION_LIMIT + (position - sell_order_volume)
        if max_sell_quantity > 0:
            orders.append(Order(self.RESIN_SYMBOL, int(round(sell_price)), -max_sell_quantity))
            sell_order_volume += max_sell_quantity
        return orders, buy_order_volume, sell_order_volume

    def rainforest_resin_strat(self, state: TradingState):
        position = state.position.get(self.RESIN_SYMBOL, 0)
        order_depth = state.order_depths[self.RESIN_SYMBOL]
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0
        take_orders, take_buy, take_sell = self.take_orders_resin(order_depth, position)
        orders.extend(take_orders)
        buy_volume += take_buy
        sell_volume += take_sell
        clear_orders, clear_buy, clear_sell = self.clear_orders_resin(order_depth, position, buy_volume, sell_volume)
        orders.extend(clear_orders)
        buy_volume += clear_buy
        sell_volume += clear_sell
        make_orders, make_buy, make_sell = self.make_orders_resin(order_depth, position, buy_volume, sell_volume)
        orders.extend(make_orders)
        return orders

    # ---------------------------
    # KELP Strategy Methods (omitted for brevity; same as previous implementations)
    # ---------------------------
    def compute_fair_value_kelp(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price, volume in order_depth.sell_orders.items() if abs(volume) >= self.KELP_ADVERSE_VOLUME]
            filtered_bid = [price for price, volume in order_depth.buy_orders.items() if abs(volume) >= self.KELP_ADVERSE_VOLUME]
            if filtered_ask and filtered_bid:
                mmmid_price = (min(filtered_ask) + max(filtered_bid)) / 2
            else:
                mmmid_price = (best_ask + best_bid) / 2 if self.kelp_last_price is None else self.kelp_last_price
            if self.kelp_last_price is not None:
                last_price = self.kelp_last_price
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.KELP_REVERSION_BETA
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            self.kelp_last_price = mmmid_price
            return fair
        return None

    def take_orders_kelp(self, order_depth: OrderDepth, position: int, fair_value: float):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            available_sell = -order_depth.sell_orders[best_ask]
            if (not self.KELP_PREVENT_ADVERSE) or (abs(available_sell) <= self.KELP_ADVERSE_VOLUME):
                if best_ask <= fair_value - self.KELP_TAKE_WIDTH:
                    max_buy = self.KELP_POSITION_LIMIT - position
                    qty = min(available_sell, max_buy)
                    if qty > 0:
                        orders.append(Order(self.KELP_SYMBOL, int(round(best_ask)), qty))
                        buy_order_volume += qty
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            available_buy = order_depth.buy_orders[best_bid]
            if (not self.KELP_PREVENT_ADVERSE) or (abs(available_buy) <= self.KELP_ADVERSE_VOLUME):
                if best_bid >= fair_value + self.KELP_TAKE_WIDTH:
                    max_sell = self.KELP_POSITION_LIMIT + position
                    qty = min(available_buy, max_sell)
                    if qty > 0:
                        orders.append(Order(self.KELP_SYMBOL, int(round(best_bid)), -qty))
                        sell_order_volume += qty
        return orders, buy_order_volume, sell_order_volume

    def clear_orders_kelp(self, order_depth: OrderDepth, position: int,
                          buy_order_volume: int, sell_order_volume: int, fair_value: float):
        orders: List[Order] = []
        net_position = position + buy_order_volume - sell_order_volume
        if net_position > 0:
            clearing_quantity = sum(volume for price, volume in order_depth.buy_orders.items()
                                    if price >= fair_value + self.KELP_CLEAR_WIDTH)
            clearing_quantity = min(clearing_quantity, net_position)
            max_sell = self.KELP_POSITION_LIMIT + position
            sell_quantity = min(max_sell, clearing_quantity)
            if sell_quantity > 0:
                orders.append(Order(self.KELP_SYMBOL, int(round(fair_value + self.KELP_CLEAR_WIDTH)), -sell_quantity))
                sell_order_volume += sell_quantity
        if net_position < 0:
            clearing_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items()
                                    if price <= fair_value - self.KELP_CLEAR_WIDTH)
            clearing_quantity = min(clearing_quantity, abs(net_position))
            max_buy = self.KELP_POSITION_LIMIT - position
            buy_quantity = min(max_buy, clearing_quantity)
            if buy_quantity > 0:
                orders.append(Order(self.KELP_SYMBOL, int(round(fair_value - self.KELP_CLEAR_WIDTH)), buy_quantity))
                buy_order_volume += buy_quantity
        return orders, buy_order_volume, sell_order_volume

    def make_orders_kelp(self, order_depth: OrderDepth, position: int,
                         buy_order_volume: int, sell_order_volume: int, fair_value: float):
        orders: List[Order] = []
        asks_above = [price for price in order_depth.sell_orders.keys() if price > fair_value + self.KELP_DISREGARD_EDGE]
        bids_below = [price for price in order_depth.buy_orders.keys() if price < fair_value - self.KELP_DISREGARD_EDGE]
        ask = int(round(fair_value + self.KELP_DEFAULT_EDGE))
        if asks_above:
            best_ask = min(asks_above)
            ask = int(round(best_ask - 1)) if abs(best_ask - fair_value) > self.KELP_JOIN_EDGE else int(round(best_ask))
        bid = int(round(fair_value - self.KELP_DEFAULT_EDGE))
        if bids_below:
            best_bid = max(bids_below)
            bid = int(round(best_bid + 1)) if abs(fair_value - best_bid) > self.KELP_JOIN_EDGE else int(round(best_bid))
        max_buy_quantity = self.KELP_POSITION_LIMIT - (position + buy_order_volume)
        if max_buy_quantity > 0:
            orders.append(Order(self.KELP_SYMBOL, bid, max_buy_quantity))
            buy_order_volume += max_buy_quantity
        max_sell_quantity = self.KELP_POSITION_LIMIT + (position - sell_order_volume)
        if max_sell_quantity > 0:
            orders.append(Order(self.KELP_SYMBOL, ask, -max_sell_quantity))
            sell_order_volume += max_sell_quantity
        return orders, buy_order_volume, sell_order_volume

    def kelp_strat(self, state: TradingState):
        product = self.KELP_SYMBOL
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        fair_value = self.compute_fair_value_kelp(order_depth)
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0
        take_orders, take_buy, take_sell = self.take_orders_kelp(order_depth, position, fair_value)
        orders.extend(take_orders)
        buy_volume += take_buy
        sell_volume += take_sell
        clear_orders, clear_buy, clear_sell = self.clear_orders_kelp(order_depth, position, buy_volume, sell_volume, fair_value)
        orders.extend(clear_orders)
        buy_volume += clear_buy
        sell_volume += clear_sell
        make_orders, make_buy, make_sell = self.make_orders_kelp(order_depth, position, buy_volume, sell_volume, fair_value)
        orders.extend(make_orders)
        return orders

    # ---------------------------
    # SQUID_INK Strategy Methods (omitted for brevity; same as previous implementations)
    # ---------------------------
    def squid_ink_strat(self, state: TradingState) -> (List[Order], str):
        product = self.SQUID_INK_SYMBOL
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        orders: List[Order] = []
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            current_price = (best_bid + best_ask) / 2
        else:
            return [], jsonpickle.encode({product: {"price_history": []}})
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"price_history": []}
        price_history = trader_data[product]["price_history"]
        price_history.append(current_price)
        if len(price_history) > self.SQUID_INK_OBSERVATION_WINDOW:
            price_history.pop(0)
        if len(price_history) >= 2:
            moving_avg = sum(price_history) / len(price_history)
            variance = sum((p - moving_avg)**2 for p in price_history) / (len(price_history) - 1)
            std_dev = math.sqrt(variance)
        else:
            moving_avg = current_price
            std_dev = 0.0001
        zscore = (current_price - moving_avg) / std_dev if std_dev != 0 else 0
        threshold = self.SQUID_INK_REVERSION_THRESHOLD
        if zscore > threshold:
            trade_price = best_bid
            max_sell = self.LIMIT[product] + position
            quantity = min(int(zscore), max_sell)
            if quantity > 0:
                orders.append(Order(product, int(round(trade_price)), -quantity))
        elif zscore < -threshold:
            trade_price = best_ask
            max_buy = self.LIMIT[product] - position
            quantity = min(int(abs(zscore)), max_buy)
            if quantity > 0:
                orders.append(Order(product, int(round(trade_price)), quantity))
        trader_data[product]["price_history"] = price_history
        traderData = jsonpickle.encode(trader_data)
        return orders, traderData

    # ---------------------------
    # NEW: VOLCANIC_ROCK Price-Based Mean-Reversion Strategy
    # ---------------------------
    def volcanic_rock_strat(self, state: TradingState) -> (List[Order], str):
        orders: List[Order] = []
        product = "VOLCANIC_ROCK"
        if product not in state.order_depths:
            print("[VolcanicRock Strat] No order depth for VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else ""
        od = state.order_depths[product]
        if not (od.buy_orders and od.sell_orders):
            print("[VolcanicRock Strat] Incomplete order book for VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else ""
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"price_history": []}
        price_history = trader_data[product]["price_history"]
        price_history.append(mid_price)
        if len(price_history) > OBSERVATION_WINDOW_COMMON:
            price_history.pop(0)
        if len(price_history) < 2:
            trader_data[product]["price_history"] = price_history
            return orders, jsonpickle.encode(trader_data)
        moving_avg = sum(price_history) / len(price_history)
        variance = sum((p - moving_avg)**2 for p in price_history) / (len(price_history) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 1e-8
        zscore = (mid_price - moving_avg) / std_dev
        print(f"[VolcanicRock Strat] mid_price: {mid_price:.2f}, moving_avg: {moving_avg:.2f}, std_dev: {std_dev:.4f}, zscore: {zscore:.2f}")
        current_pos = state.position.get(product, 0)
        max_buy = self.VOLCANIC_ROCK_POSITION_LIMIT - current_pos
        max_sell = self.VOLCANIC_ROCK_POSITION_LIMIT + current_pos
        if zscore > 2:
            qty = min(int(zscore), max_sell)
            if qty > 0:
                print(f"[VolcanicRock Strat] SELL signal: best_bid: {best_bid}, qty: {qty}")
                orders.append(Order(product, int(round(best_bid)), -qty))
        elif zscore < -2:
            qty = min(int(abs(zscore)), max_buy)
            if qty > 0:
                print(f"[VolcanicRock Strat] BUY signal: best_ask: {best_ask}, qty: {qty}")
                orders.append(Order(product, int(round(best_ask)), qty))
        else:
            print("[VolcanicRock Strat] No trading signal; not trading.")
        trader_data[product]["price_history"] = price_history
        updated_traderData = jsonpickle.encode(trader_data)
        return orders, updated_traderData

    # ---------------------------
    # NEW: Voucher Vega Mean-Reversion Strategy (for strikes 10000, 10250, 10500)
    # ---------------------------
    def voucher_vega_strat(self, state: TradingState, voucher_symbol: str, strike: int) -> (List[Order], str, float):
        orders: List[Order] = []
        product = voucher_symbol
        delta_exposure = 0.0
        if product not in state.order_depths:
            print(f"[Voucher Vega] No order depth for {product}.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        od = state.order_depths[product]
        if not (od.buy_orders and od.sell_orders):
            print(f"[Voucher Vega] Incomplete order book for {product}.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        underlying_sym = "VOLCANIC_ROCK"
        if underlying_sym not in state.order_depths:
            print("[Voucher Vega] No order depth for underlying VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        underlying_mid = get_mid_price(state.order_depths[underlying_sym])
        if underlying_mid is None or underlying_mid <= 0:
            print("[Voucher Vega] Invalid underlying mid-price.")
            return orders, state.traderData if state.traderData else "", delta_exposure

        day = getattr(state, "day", 0)
        timestamp = getattr(state, "timestamp", 0)
        TTE = compute_tte(day, timestamp)
        if TTE <= 0:
            print("[Voucher Vega] Non-positive TTE.")
            return orders, state.traderData if state.traderData else "", delta_exposure

        v = implied_volatility_bisection(underlying_mid, mid_price, strike, TTE)
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"iv_history": []}
        iv_history = trader_data[product]["iv_history"]
        iv_history.append(v)
        if len(iv_history) > OBSERVATION_WINDOW_COMMON:
            iv_history.pop(0)
        if len(iv_history) < 2:
            trader_data[product]["iv_history"] = iv_history
            return orders, jsonpickle.encode(trader_data), delta_exposure
        mean_iv = sum(iv_history) / len(iv_history)
        variance_iv = sum((iv - mean_iv)**2 for iv in iv_history) / (len(iv_history) - 1)
        std_iv = math.sqrt(variance_iv) if variance_iv > 0 else 1e-8
        zscore = (v - mean_iv) / std_iv
        print(f"[Voucher Vega] {product} mid: {mid_price:.2f}, IV: {v:.4f}, mean IV: {mean_iv:.4f}, std IV: {std_iv:.4f}, zscore: {zscore:.2f}")
        current_pos = state.position.get(product, 0)
        max_buy = self.VOUCHER_POSITION_LIMIT - current_pos
        max_sell = self.VOUCHER_POSITION_LIMIT + current_pos
        if zscore > self.voucher_zscore_threshold:
            qty = min(int(zscore) * self.voucher_scaler, max_sell)
            if qty > 0:
                print(f"[Voucher Vega] SELL signal for {product}: best_bid: {best_bid}, scaled qty: {qty}")
                orders.append(Order(product, int(round(best_bid)), -qty))
                d1 = (math.log(underlying_mid/strike) + 0.5*v*v*TTE) / (v*math.sqrt(TTE))
                delta = norm_cdf(d1)
                delta_exposure = -qty * delta
        elif zscore < -self.voucher_zscore_threshold:
            qty = min(int(abs(zscore)) * self.voucher_scaler, max_buy)
            if qty > 0:
                print(f"[Voucher Vega] BUY signal for {product}: best_ask: {best_ask}, scaled qty: {qty}")
                orders.append(Order(product, int(round(best_ask)), qty))
                d1 = (math.log(underlying_mid/strike) + 0.5*v*v*TTE) / (v*math.sqrt(TTE))
                delta = norm_cdf(d1)
                delta_exposure = qty * delta
        else:
            print(f"[Voucher Vega] {product}: No strong trading signal; not trading.")
        trader_data[product]["iv_history"] = iv_history
        updated_traderData = jsonpickle.encode(trader_data)
        return orders, updated_traderData, delta_exposure

    def hedge_delta(self, state: TradingState, net_delta: float,
                    underlying_sym: str, voucher9500_sym: str, voucher9750_sym: str) -> List[Order]:
        hedge_orders = []
        underlying_pos = state.position.get(underlying_sym, 0)
        voucher9500_pos = state.position.get(voucher9500_sym, 0)
        voucher9750_pos = state.position.get(voucher9750_sym, 0)
        underlying_limit = 400
        voucher_limit = 200
        remaining = abs(net_delta)
        if net_delta > 0:
            available_underlying = underlying_limit - underlying_pos
            if available_underlying > 0:
                hedge_qty = min(remaining, available_underlying)
                od = state.order_depths[underlying_sym]
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    hedge_orders.append(Order(underlying_sym, int(round(best_bid)), -int(round(hedge_qty))))
                    remaining -= hedge_qty
            if remaining > 0:
                available_v9500 = voucher_limit - voucher9500_pos
                if available_v9500 > 0:
                    hedge_qty = min(remaining, available_v9500)
                    od = state.order_depths.get(voucher9500_sym, None)
                    if od and od.buy_orders:
                        best_bid = max(od.buy_orders.keys())
                        hedge_orders.append(Order(voucher9500_sym, int(round(best_bid)), -int(round(hedge_qty))))
                        remaining -= hedge_qty
            if remaining > 0:
                available_v9750 = voucher_limit - voucher9750_pos
                if available_v9750 > 0:
                    hedge_qty = min(remaining, available_v9750)
                    od = state.order_depths.get(voucher9750_sym, None)
                    if od and od.buy_orders:
                        best_bid = max(od.buy_orders.keys())
                        hedge_orders.append(Order(voucher9750_sym, int(round(best_bid)), -int(round(hedge_qty))))
                        remaining -= hedge_qty
        elif net_delta < 0:
            remaining = abs(net_delta)
            available_underlying = underlying_limit + underlying_pos
            if available_underlying > 0:
                hedge_qty = min(remaining, available_underlying)
                od = state.order_depths[underlying_sym]
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    hedge_orders.append(Order(underlying_sym, int(round(best_ask)), int(round(hedge_qty))))
                    remaining -= hedge_qty
            if remaining > 0:
                available_v9500 = voucher_limit + voucher9500_pos
                if available_v9500 > 0:
                    hedge_qty = min(remaining, available_v9500)
                    od = state.order_depths.get(voucher9500_sym, None)
                    if od and od.sell_orders:
                        best_ask = min(od.sell_orders.keys())
                        hedge_orders.append(Order(voucher9500_sym, int(round(best_ask)), int(round(hedge_qty))))
                        remaining -= hedge_qty
            if remaining > 0:
                available_v9750 = voucher_limit + voucher9750_pos
                if available_v9750 > 0:
                    hedge_qty = min(remaining, available_v9750)
                    od = state.order_depths.get(voucher9750_sym, None)
                    if od and od.sell_orders:
                        best_ask = min(od.sell_orders.keys())
                        hedge_orders.append(Order(voucher9750_sym, int(round(best_ask)), int(round(hedge_qty))))
                        remaining -= hedge_qty
        return hedge_orders

    # ---------------------------
    # SQUID_INK Strategy Methods (omitted for brevity; same as previous implementations)
    # ---------------------------
    def squid_ink_strat(self, state: TradingState) -> (List[Order], str):
        product = self.SQUID_INK_SYMBOL
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        orders: List[Order] = []
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            current_price = (best_bid + best_ask) / 2
        else:
            return [], jsonpickle.encode({product: {"price_history": []}})
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"price_history": []}
        price_history = trader_data[product]["price_history"]
        price_history.append(current_price)
        if len(price_history) > self.SQUID_INK_OBSERVATION_WINDOW:
            price_history.pop(0)
        if len(price_history) >= 2:
            moving_avg = sum(price_history) / len(price_history)
            variance = sum((p - moving_avg)**2 for p in price_history) / (len(price_history) - 1)
            std_dev = math.sqrt(variance)
        else:
            moving_avg = current_price
            std_dev = 0.0001
        zscore = (current_price - moving_avg) / std_dev if std_dev != 0 else 0
        threshold = self.SQUID_INK_REVERSION_THRESHOLD
        if zscore > threshold:
            trade_price = best_bid
            max_sell = self.LIMIT[product] + position
            quantity = min(int(zscore), max_sell)
            if quantity > 0:
                orders.append(Order(product, int(round(trade_price)), -quantity))
        elif zscore < -threshold:
            trade_price = best_ask
            max_buy = self.LIMIT[product] - position
            quantity = min(int(abs(zscore)), max_buy)
            if quantity > 0:
                orders.append(Order(product, int(round(trade_price)), quantity))
        trader_data[product]["price_history"] = price_history
        traderData = jsonpickle.encode(trader_data)
        return orders, traderData

    # ---------------------------
    # NEW: VOLCANIC_ROCK Price-Based Mean-Reversion Strategy
    # ---------------------------
    def volcanic_rock_strat(self, state: TradingState) -> (List[Order], str):
        orders: List[Order] = []
        product = "VOLCANIC_ROCK"
        if product not in state.order_depths:
            print("[VolcanicRock Strat] No order depth for VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else ""
        od = state.order_depths[product]
        if not (od.buy_orders and od.sell_orders):
            print("[VolcanicRock Strat] Incomplete order book for VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else ""
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"price_history": []}
        price_history = trader_data[product]["price_history"]
        price_history.append(mid_price)
        if len(price_history) > OBSERVATION_WINDOW_COMMON:
            price_history.pop(0)
        if len(price_history) < 2:
            trader_data[product]["price_history"] = price_history
            return orders, jsonpickle.encode(trader_data)
        moving_avg = sum(price_history) / len(price_history)
        variance = sum((p - moving_avg)**2 for p in price_history) / (len(price_history) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 1e-8
        zscore = (mid_price - moving_avg) / std_dev
        print(f"[VolcanicRock Strat] mid_price: {mid_price:.2f}, moving_avg: {moving_avg:.2f}, std_dev: {std_dev:.4f}, zscore: {zscore:.2f}")
        current_pos = state.position.get(product, 0)
        max_buy = self.VOLCANIC_ROCK_POSITION_LIMIT - current_pos
        max_sell = self.VOLCANIC_ROCK_POSITION_LIMIT + current_pos
        if zscore > 2:
            qty = min(int(zscore), max_sell)
            if qty > 0:
                print(f"[VolcanicRock Strat] SELL signal: best_bid: {best_bid}, qty: {qty}")
                orders.append(Order(product, int(round(best_bid)), -qty))
        elif zscore < -2:
            qty = min(int(abs(zscore)), max_buy)
            if qty > 0:
                print(f"[VolcanicRock Strat] BUY signal: best_ask: {best_ask}, qty: {qty}")
                orders.append(Order(product, int(round(best_ask)), qty))
        else:
            print("[VolcanicRock Strat] No trading signal; not trading.")
        trader_data[product]["price_history"] = price_history
        updated_traderData = jsonpickle.encode(trader_data)
        return orders, updated_traderData

    # ---------------------------
    # NEW: Voucher Vega Mean-Reversion Strategy (for strikes 10000, 10250, 10500)
    # ---------------------------
    def voucher_vega_strat(self, state: TradingState, voucher_symbol: str, strike: int) -> (List[Order], str, float):
        orders: List[Order] = []
        product = voucher_symbol
        delta_exposure = 0.0
        if product not in state.order_depths:
            print(f"[Voucher Vega] No order depth for {product}.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        od = state.order_depths[product]
        if not (od.buy_orders and od.sell_orders):
            print(f"[Voucher Vega] Incomplete order book for {product}.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        underlying_sym = "VOLCANIC_ROCK"
        if underlying_sym not in state.order_depths:
            print("[Voucher Vega] No order depth for underlying VOLCANIC_ROCK.")
            return orders, state.traderData if state.traderData else "", delta_exposure
        underlying_mid = get_mid_price(state.order_depths[underlying_sym])
        if underlying_mid is None or underlying_mid <= 0:
            print("[Voucher Vega] Invalid underlying mid-price.")
            return orders, state.traderData if state.traderData else "", delta_exposure

        day = getattr(state, "day", 0)
        timestamp = getattr(state, "timestamp", 0)
        TTE = compute_tte(day, timestamp)
        if TTE <= 0:
            print("[Voucher Vega] Non-positive TTE.")
            return orders, state.traderData if state.traderData else "", delta_exposure

        v = implied_volatility_bisection(underlying_mid, mid_price, strike, TTE)
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        if product not in trader_data:
            trader_data[product] = {"iv_history": []}
        iv_history = trader_data[product]["iv_history"]
        iv_history.append(v)
        if len(iv_history) > OBSERVATION_WINDOW_COMMON:
            iv_history.pop(0)
        if len(iv_history) < 2:
            trader_data[product]["iv_history"] = iv_history
            return orders, jsonpickle.encode(trader_data), delta_exposure
        mean_iv = sum(iv_history) / len(iv_history)
        variance_iv = sum((iv - mean_iv) ** 2 for iv in iv_history) / (len(iv_history) - 1)
        std_iv = math.sqrt(variance_iv) if variance_iv > 0 else 1e-8
        zscore = (v - mean_iv) / std_iv
        print(f"[Voucher Vega] {product} mid: {mid_price:.2f}, IV: {v:.4f}, mean IV: {mean_iv:.4f}, std IV: {std_iv:.4f}, zscore: {zscore:.2f}")
        current_pos = state.position.get(product, 0)
        max_buy = self.VOUCHER_POSITION_LIMIT - current_pos
        max_sell = self.VOUCHER_POSITION_LIMIT + current_pos
        if zscore > self.voucher_zscore_threshold:
            qty = min(int(zscore) * self.voucher_scaler, max_sell)
            if qty > 0:
                print(f"[Voucher Vega] SELL signal for {product}: best_bid: {best_bid}, scaled qty: {qty}")
                orders.append(Order(product, int(round(best_bid)), -qty))
                d1 = (math.log(underlying_mid/strike) + 0.5*v*v*TTE) / (v*math.sqrt(TTE))
                delta = norm_cdf(d1)
                delta_exposure = -qty * delta
        elif zscore < -self.voucher_zscore_threshold:
            qty = min(int(abs(zscore)) * self.voucher_scaler, max_buy)
            if qty > 0:
                print(f"[Voucher Vega] BUY signal for {product}: best_ask: {best_ask}, scaled qty: {qty}")
                orders.append(Order(product, int(round(best_ask)), qty))
                d1 = (math.log(underlying_mid/strike) + 0.5*v*v*TTE) / (v*math.sqrt(TTE))
                delta = norm_cdf(d1)
                delta_exposure = qty * delta
        else:
            print(f"[Voucher Vega] {product}: No strong trading signal; not trading.")
        trader_data[product]["iv_history"] = iv_history
        updated_traderData = jsonpickle.encode(trader_data)
        return orders, updated_traderData, delta_exposure

    def hedge_delta(self, state: TradingState, net_delta: float,
                    underlying_sym: str, voucher9500_sym: str, voucher9750_sym: str) -> List[Order]:
        hedge_orders = []
        underlying_pos = state.position.get(underlying_sym, 0)
        voucher9500_pos = state.position.get(voucher9500_sym, 0)
        voucher9750_pos = state.position.get(voucher9750_sym, 0)
        underlying_limit = 400
        voucher_limit = 200
        remaining = abs(net_delta)
        if net_delta > 0:
            available_underlying = underlying_limit - underlying_pos
            if available_underlying > 0:
                hedge_qty = min(remaining, available_underlying)
                od = state.order_depths[underlying_sym]
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    hedge_orders.append(Order(underlying_sym, int(round(best_bid)), -int(round(hedge_qty))))
                    remaining -= hedge_qty
            if remaining > 0:
                available_v9500 = voucher_limit - voucher9500_pos
                if available_v9500 > 0:
                    hedge_qty = min(remaining, available_v9500)
                    od = state.order_depths.get(voucher9500_sym, None)
                    if od and od.buy_orders:
                        best_bid = max(od.buy_orders.keys())
                        hedge_orders.append(Order(voucher9500_sym, int(round(best_bid)), -int(round(hedge_qty))))
                        remaining -= hedge_qty
            if remaining > 0:
                available_v9750 = voucher_limit - voucher9750_pos
                if available_v9750 > 0:
                    hedge_qty = min(remaining, available_v9750)
                    od = state.order_depths.get(voucher9750_sym, None)
                    if od and od.buy_orders:
                        best_bid = max(od.buy_orders.keys())
                        hedge_orders.append(Order(voucher9750_sym, int(round(best_bid)), -int(round(hedge_qty))))
                        remaining -= hedge_qty
        elif net_delta < 0:
            remaining = abs(net_delta)
            available_underlying = underlying_limit + underlying_pos
            if available_underlying > 0:
                hedge_qty = min(remaining, available_underlying)
                od = state.order_depths[underlying_sym]
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    hedge_orders.append(Order(underlying_sym, int(round(best_ask)), int(round(hedge_qty))))
                    remaining -= hedge_qty
            if remaining > 0:
                available_v9500 = voucher_limit + voucher9500_pos
                if available_v9500 > 0:
                    hedge_qty = min(remaining, available_v9500)
                    od = state.order_depths.get(voucher9500_sym, None)
                    if od and od.sell_orders:
                        best_ask = min(od.sell_orders.keys())
                        hedge_orders.append(Order(voucher9500_sym, int(round(best_ask)), int(round(hedge_qty))))
                        remaining -= hedge_qty
            if remaining > 0:
                available_v9750 = voucher_limit + voucher9750_pos
                if available_v9750 > 0:
                    hedge_qty = min(remaining, available_v9750)
                    od = state.order_depths.get(voucher9750_sym, None)
                    if od and od.sell_orders:
                        best_ask = min(od.sell_orders.keys())
                        hedge_orders.append(Order(voucher9750_sym, int(round(best_ask)), int(round(hedge_qty))))
                        remaining -= hedge_qty
        return hedge_orders

class BasketTrader:
    def __init__(self, basket_symbol: str, synthetic_weights: dict, baseline_mean: float,
                 window: int, zscore_threshold: float, target_position: int):
        self.basket_symbol = basket_symbol
        self.synthetic_weights = synthetic_weights
        self.baseline_mean = baseline_mean
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.target_position = target_position
        self.spread_history = []

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def get_synthetic_mid(self, state: TradingState) -> float:
        synthetic_mid = 0
        for product, weight in self.synthetic_weights.items():
            if product not in state.order_depths:
                return None
            mid = self.get_mid_price(state.order_depths[product])
            if mid is None:
                return None
            synthetic_mid += weight * mid
        return synthetic_mid

    def update(self, state: TradingState, result: dict):
        if self.basket_symbol not in state.order_depths:
            return None
        basket_od = state.order_depths[self.basket_symbol]
        basket_mid = self.get_mid_price(basket_od)
        if basket_mid is None:
            return None
        synthetic_mid = self.get_synthetic_mid(state)
        if synthetic_mid is None:
            return None
        spread = basket_mid - synthetic_mid
        self.spread_history.append(spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)
        if len(self.spread_history) < self.window:
            return None
        spread_std = np.std(self.spread_history)
        if spread_std == 0:
            spread_std = 1e-8
        zscore = (spread - self.baseline_mean) / spread_std
        current_position = state.position.get(self.basket_symbol, 0)
        order_size_cap = 20
        scaling_factor = 1
        if zscore > self.zscore_threshold and current_position > -self.target_position:
            quantity = min(current_position + self.target_position, int(round(order_size_cap * scaling_factor)))
            if self.basket_symbol not in result:
                result[self.basket_symbol] = []
            result[self.basket_symbol].append(Order(self.basket_symbol, int(basket_mid), -quantity))
            for product, weight in self.synthetic_weights.items():
                if product not in result:
                    result[product] = []
                result[product].append(Order(product, int(self.get_mid_price(state.order_depths[product])),
                                            int(quantity * weight * 250 / (self.synthetic_weights["CROISSANTS"] * 160))))
        elif zscore < -self.zscore_threshold and current_position < self.target_position:
            quantity = min(self.target_position - current_position, int(round(order_size_cap * scaling_factor)))
            if self.basket_symbol not in result:
                result[self.basket_symbol] = []
            result[self.basket_symbol].append(Order(self.basket_symbol, int(basket_mid), quantity))
            for product, weight in self.synthetic_weights.items():
                if product not in result:
                    result[product] = []
                result[product].append(Order(product, int(self.get_mid_price(state.order_depths[product])),
                                            int(-quantity * weight * 250 / (self.synthetic_weights["CROISSANTS"] * 160))))
        print(f"[{self.basket_symbol}] basket_mid = {basket_mid:.2f}, synthetic_mid = {synthetic_mid:.2f}, "
              f"spread = {spread:.2f}, std = {spread_std:.2f}, zscore = {zscore:.2f}, position = {current_position}")
        return result

