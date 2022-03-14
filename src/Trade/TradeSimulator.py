from typing import List
import pandas as pd
import numpy as np
from mltradeshared.Trade import Trade, TradeManager
from mltradeshared import RNN

class TradeSimulator:
    def __init__(self, val_df: pd.DataFrame, val_x: np.ndarray, tm: TradeManager, rnn: RNN, atr_period = 14):
        """
        val_df should be the raw data, but the bit that refers to the validation data
        val_x should be the preprocessed data
        """
        if len(val_df) < len(val_x) + atr_period:
            raise Exception("val_df must be longer than val_x + atr_period. This class will take the last len(val_x) values in val_df for simulation (and needs an extra atr_period to calc the atr)")        
        self.atr_series = self._get_ATR_series(val_df, atr_period).iloc[-len(val_x):]
        self.val_df: pd.DataFrame = val_df.iloc[-len(val_x):]
        self.val_df.reset_index(drop=True, inplace=True)
        self.atr_series.reset_index(drop=True, inplace=True)
        self.val_x = val_x
        self.tm = tm
        self.rnn = rnn

    def _get_ATR_series(self, data: pd.DataFrame, period: int) -> pd.Series:
        high_low = data['h'] - data['l']
        high_cp = np.abs(data['h'] - data['c'].shift())
        low_cp = np.abs(data['l'] - data['c'].shift())

        df = pd.concat([high_low, high_cp, low_cp], axis=1)
        true_range = np.max(df, axis=1)

        average_true_range = true_range.rolling(period).mean()
        return average_true_range

    def start(self):

        def set_trade_id(trade: Trade):
            trade.ticket_id = 0 
            print(f"FILLED {trade.ticket_id}. ({'BUY' if trade.is_buy else 'SELL'})")

        def close_trades(trades: List[Trade]):
            for trade in trades:
                print(f"CLOSED {trade.ticket_id}. ({'BUY' if trade.is_buy else 'SELL'}) --- Balance: {self.tm.balance}")
        
        predictions = self.rnn.model.predict(self.val_x)
        for index, row in self.val_df.iterrows():
            data_point = row.to_dict()
            prediction = predictions[index]
            self.tm.check_open_trades(data_point, close_trades)
            if self.tm.should_make_trade(prediction, data_point):
                ATR = self.atr_series.iloc[index]
                self.tm.make_trade(prediction, data_point, set_trade_id, ATR=ATR)

    def summary(self):
        final_balance = self.tm.balance
        average_loss_pct = self.tm.risk_per_trade * 100
        average_win_pct = self.tm.risk_per_trade * (self.tm.take_profit_ATR / self.tm.stop_loss_ATR) * 100
        num_trades = 0
        num_wins = 0
        num_losses = 0
        for trade in self.tm.closed_trades:
            if trade.close_price > trade.open_price:
                if trade.is_buy:
                    num_wins += 1
                else:
                    num_losses += 1
            else:
                if not trade.is_buy:
                    num_wins += 1
                else:
                    num_losses += 1
            num_trades += 1

        print(f"\nFinal Balance: {final_balance}")
        print(f"Number of trades: {num_trades}")
        print(f"Number of wins: {num_wins}")
        print(f"Number of losses: {num_losses}")
        print(f"Average Win: {average_win_pct}%")
        print(f"Average Loss: {average_loss_pct}%")