import pandas as pd
import json
from datetime import timedelta
import logging
from collections import Counter

class PerformanceAnalyzer:
    """
    Analyzes simulation logs to correlate performance with news events and
    generates a set of statistically-backed, learned rules based on the findings.
    """
    def __init__(self, equity_log_path, trade_log_path, news_log_path, vix_log_path):
        try:
            self.equity_log = pd.read_csv(equity_log_path, parse_dates=['Timestamp'], index_col='Timestamp')
        except FileNotFoundError: self.equity_log = pd.DataFrame()

        try:
            self.trade_log = pd.read_csv(trade_log_path, parse_dates=['Timestamp'])
        except FileNotFoundError: self.trade_log = pd.DataFrame()

        try:
            with open(news_log_path, 'r') as f:
                news_data = json.load(f)
                self.news_log = pd.DataFrame(news_data) if news_data else pd.DataFrame()
                if not self.news_log.empty:
                    self.news_log['created_at'] = pd.to_datetime(self.news_log['created_at']).dt.tz_localize(None)
        except (json.JSONDecodeError, FileNotFoundError): self.news_log = pd.DataFrame()
        
        try:
            self.vix_log = pd.read_csv(vix_log_path, parse_dates=['Date'], index_col='Date')
        except FileNotFoundError: self.vix_log = pd.DataFrame()


    def run_analysis(self):
        """
        Identifies the best/worst days, runs the news analysis, and then
        generates a set of learned rules from the results.
        """
        if self.equity_log.empty or len(self.equity_log) < 2:
            return {'error': 'Equity log has insufficient data to perform analysis.'}

        self.equity_log['profit_loss'] = self.equity_log['Equity'].diff()
        sorted_days = self.equity_log.sort_values(by='profit_loss', ascending=False).dropna()
        
        # --- THIS IS THE FIX ---
        # Gracefully handle cases with few data points
        num_days = len(sorted_days)
        if num_days == 0:
            return {'best_days': [], 'worst_days': [], 'learned_rules': []}

        num_best = min(3, num_days // 2)
        num_worst = min(3, num_days - num_best)

        best_days = sorted_days.head(num_best) if num_best > 0 else pd.DataFrame()
        worst_days = sorted_days.tail(num_worst) if num_worst > 0 else pd.DataFrame()

        analysis = {
            'best_days': self._analyze_days(best_days),
            'worst_days': self._analyze_days(worst_days)
        }

        learned_rules = self._learn_from_analysis()
        analysis['learned_rules'] = learned_rules

        return analysis

    def _analyze_days(self, days_df):
        """
        Analyzes a set of days by finding both directly relevant news and major
        market-wide news events.
        """
        if days_df.empty:
            return []
            
        results = []
        market_keywords = [
            'fed', 'rate', 'inflation', 'gdp', 'unemployment', 'cpi', 'jobs report',
            'war', 'crisis', 'election', 'trade deal', 'tariff', 'geopolitical',
            'market crash', 'rally', 'correction', 'volatility', 'earnings', 'guidance',
            'supply chain', 'stimulus'
        ]

        for date, row in days_df.iterrows():
            day_str = date.strftime('%Y-%m-%d')
            
            trades_on_day = self.trade_log[self.trade_log['Timestamp'] == date] if not self.trade_log.empty else pd.DataFrame()
            symbols_traded = trades_on_day['Symbol'].unique().tolist()
            
            specific_news_df, market_news_df = pd.DataFrame(), pd.DataFrame()

            if not self.news_log.empty:
                news_window_start = date - timedelta(days=1)
                relevant_time_news = self.news_log[
                    (self.news_log['created_at'] >= news_window_start) & 
                    (self.news_log['created_at'] < date + timedelta(days=1))
                ]

                if not relevant_time_news.empty:
                    if symbols_traded:
                        specific_news_df = relevant_time_news[
                            relevant_time_news['symbols'].apply(lambda symbols_list: any(s in symbols_list for s in symbols_traded))
                        ]
                    
                    market_news_df = relevant_time_news[
                        relevant_time_news['snippet'].str.contains('|'.join(market_keywords), case=False, na=False)
                    ]
                    
                    if not specific_news_df.empty and not market_news_df.empty:
                        market_news_df = market_news_df[~market_news_df.index.isin(specific_news_df.index)]

            vix_context = "N/A"
            if not self.vix_log.empty:
                try:
                    vix_val = self.vix_log.loc[day_str]['Close']
                    if vix_val > 30: vix_context = f"High Volatility / Panic (VIX: {vix_val:.2f})"
                    elif vix_val < 15: vix_context = f"Low Volatility / Complacency (VIX: {vix_val:.2f})"
                    else: vix_context = f"Normal Volatility (VIX: {vix_val:.2f})"
                except KeyError:
                    pass

            results.append({
                'date': day_str,
                'profit_loss': f"${row['profit_loss']:,.2f}",
                'trades': trades_on_day.to_dict('records'),
                'news': {
                    'specific': specific_news_df.to_dict('records'),
                    'market': market_news_df.to_dict('records')
                },
                'vix_context': vix_context
            })
        return results

    def _learn_from_analysis(self):
        """
        Scans the entire simulation history to find keywords with a strong statistical
        correlation to negative returns, then generates defensive rules.
        """
        market_keywords = [
            'fed', 'rate', 'inflation', 'gdp', 'unemployment', 'cpi', 'jobs report',
            'war', 'crisis', 'election', 'trade deal', 'tariff', 'geopolitical',
            'market crash', 'correction', 'volatility', 'downgrade', 'guidance',
            'supply chain', 'stimulus', 'recession'
        ]
        
        keyword_performance = {kw: [] for kw in market_keywords}

        if self.news_log.empty or self.equity_log.empty:
            return []

        for date, row in self.equity_log.iterrows():
            daily_pnl = row['profit_loss']
            if pd.isna(daily_pnl): continue

            news_window_start = date - timedelta(days=1)
            relevant_news = self.news_log[
                (self.news_log['created_at'] >= news_window_start) & 
                (self.news_log['created_at'] < date + timedelta(days=1))
            ]

            if not relevant_news.empty:
                unique_keywords_in_day = set()
                for headline in relevant_news['headline']:
                    for keyword in market_keywords:
                        if keyword in headline.lower():
                            unique_keywords_in_day.add(keyword)
                
                for keyword in unique_keywords_in_day:
                    keyword_performance[keyword].append(daily_pnl)

        learned_rules = []
        initial_equity = self.equity_log['Equity'].iloc[0]

        for keyword, pnl_list in keyword_performance.items():
            if len(pnl_list) >= 5:
                avg_pnl = sum(pnl_list) / len(pnl_list)
                if avg_pnl < -(initial_equity * 0.001):
                    rule = {
                        "trigger_keyword": keyword,
                        "action": "enter_heightened_alert",
                        "duration_minutes": 120,
                        "confidence": f"High ({len(pnl_list)} occurrences, avg loss ${avg_pnl:,.2f})"
                    }
                    learned_rules.append(rule)
        
        try:
            with open('learned_rules.json', 'w') as f:
                json.dump(learned_rules, f, indent=2)
            logging.info(f"Successfully saved {len(learned_rules)} new rules to learned_rules.json")
        except Exception as e:
            logging.error(f"Failed to save learned rules: {e}")

        return learned_rules

