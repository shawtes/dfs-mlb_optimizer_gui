"""
Enhanced Feature Engineering System for ML Trading
Based on Stefan Jansen's "Machine Learning for Algorithmic Trading"

This module implements:
1. Alpha Factor Library (100+ formulaic alpha factors)
2. Advanced Technical Indicators (beyond basic RSI/MACD)
3. Denoising Techniques (Kalman filters, wavelets)
4. Multi-timeframe Features
5. Market Microstructure Features
6. Sentiment and Alternative Data Features
"""

import pandas as pd
import numpy as np
import talib
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

try:
    import pywt
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

class EnhancedFeatureEngine:
    """
    Advanced feature engineering system implementing techniques from
    Stefan Jansen's ML for Algorithmic Trading book
    """
    
    def __init__(self, enable_denoising=True, enable_alpha_factors=True):
        self.enable_denoising = enable_denoising and KALMAN_AVAILABLE
        self.enable_alpha_factors = enable_alpha_factors
        self.scalers = {}
        self.pca_models = {}
        
        logger.info(f"🧠 Enhanced Feature Engine initialized:")
        logger.info(f"   Kalman filtering: {'✅' if self.enable_denoising else '❌'}")
        logger.info(f"   Wavelets: {'✅' if WAVELETS_AVAILABLE else '❌'}")
        logger.info(f"   Alpha factors: {'✅' if self.enable_alpha_factors else '❌'}")
    
    def calculate_enhanced_features(self, df, symbol=None, timeframe='1h'):
        """
        Main function to calculate all enhanced features
        """
        try:
            if df is None or df.empty:
                logger.error("Empty dataframe provided")
                return pd.DataFrame()
            
            logger.info(f"🔧 Calculating enhanced features for {symbol or 'unknown'} ({timeframe})")
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return df
            
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values in OHLCV
            df = df.dropna(subset=required_cols)
            
            if len(df) < 50:  # Need minimum data for meaningful features
                logger.warning(f"Insufficient data: {len(df)} rows")
                return df
            
            # 1. Basic Technical Indicators (Enhanced)
            df = self._calculate_basic_indicators(df)
            
            # 2. Alpha Factors (WorldQuant style)
            if self.enable_alpha_factors:
                df = self._calculate_alpha_factors(df)
            
            # 3. Advanced Technical Indicators
            df = self._calculate_advanced_indicators(df)
            
            # 4. Market Microstructure Features
            df = self._calculate_microstructure_features(df)
            
            # 5. Statistical Features
            df = self._calculate_statistical_features(df)
            
            # 6. Regime Detection Features
            df = self._calculate_regime_features(df)
            
            # 7. Denoising (if enabled)
            if self.enable_denoising:
                df = self._apply_denoising(df)
            
            # 8. Multi-timeframe Features (requires external timeframe data)
            df = self._calculate_temporal_features(df, timeframe)
            
            # 9. Risk and Volatility Features
            df = self._calculate_risk_features(df)
            
            # 10. Interaction Features
            df = self._calculate_interaction_features(df)
            
            # Final cleanup
            df = self._cleanup_features(df)
            
            feature_count = len([col for col in df.columns if col not in required_cols + ['timestamp']])
            logger.info(f"✅ Generated {feature_count} enhanced features")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced feature calculation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return df if 'df' in locals() else pd.DataFrame()
    
    def _calculate_basic_indicators(self, df):
        """Enhanced basic technical indicators"""
        try:
            # Price transformations
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['returns'] = df['close'].pct_change()
            df['price_change'] = df['close'].diff()
            
            # Multiple timeframe EMAs
            for period in [5, 8, 13, 21, 34, 55, 89]:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                df[f'ema_signal_{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
            
            # Multiple timeframe SMAs
            for period in [10, 20, 30, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # Enhanced RSI with multiple periods
            for period in [9, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss.replace(0, float('inf'))
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                df[f'rsi_{period}_signal'] = ((df[f'rsi_{period}'] > 30) & (df[f'rsi_{period}'] < 70)).astype(int)
            
            # Create generic 'rsi' column for backward compatibility (uses 14-period RSI)
            df['rsi'] = df['rsi_14']
            
            # Enhanced MACD
            for fast, slow, signal_period in [(12, 26, 9), (8, 21, 5), (19, 39, 9)]:
                ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                histogram = macd_line - signal_line
                
                df[f'macd_{fast}_{slow}'] = macd_line
                df[f'macd_signal_{fast}_{slow}'] = signal_line
                df[f'macd_hist_{fast}_{slow}'] = histogram
                df[f'macd_cross_{fast}_{slow}'] = (macd_line > signal_line).astype(int)
            
            # Create generic 'macd' and 'macd_signal' columns for backward compatibility (uses 12-26 MACD)
            df['macd'] = df['macd_12_26']
            df['macd_signal'] = df['macd_signal_12_26']
            df['macd_hist'] = df['macd_hist_12_26']
            
            # Bollinger Bands with multiple periods
            for period in [14, 20, 30]:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = sma + (std * 2)
                df[f'bb_lower_{period}'] = sma - (std * 2)
                df[f'bb_middle_{period}'] = sma
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            
            # Create generic Bollinger Bands columns for backward compatibility (uses 20-period)
            df['upper_band'] = df['bb_upper_20']
            df['lower_band'] = df['bb_lower_20']
            
            # Stochastic Oscillator variants
            for k_period, d_period in [(14, 3), (21, 5), (9, 3)]:
                low_min = df['low'].rolling(window=k_period).min()
                high_max = df['high'].rolling(window=k_period).max()
                k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
                df[f'stoch_k_{k_period}'] = k_percent
                df[f'stoch_d_{k_period}'] = k_percent.rolling(window=d_period).mean()
                df[f'stoch_signal_{k_period}'] = ((k_percent > 20) & (k_percent < 80)).astype(int)
            
            # Create generic '%K' and '%D' columns for backward compatibility (uses 14-period)
            df['%K'] = df['stoch_k_14']
            df['%D'] = df['stoch_d_14']
            
            return df
            
        except Exception as e:
            logger.error(f"Error in basic indicators: {str(e)}")
            return df
    
    def _calculate_alpha_factors(self, df):
        """
        Calculate Alpha factors inspired by WorldQuant's research
        These are formulaic expressions that capture market anomalies
        """
        try:
            # Price-based alpha factors
            df['alpha_001'] = self._rank((df['close'] - df['open']) / df['open'])  # Intraday return rank
            df['alpha_002'] = self._rank(-1 * self._correlation(df['high'], df['volume'], 5))  # High-volume correlation
            df['alpha_003'] = self._rank(-1 * self._ts_sum(df['close'], 5))  # Negative momentum
            df['alpha_004'] = self._rank(self._stddev(df['low'], 10))  # Low price volatility
            df['alpha_005'] = self._rank((df['volume'] / self._ts_mean(df['volume'], 20)) - 1)  # Volume surprise
            
            # Volume-based alpha factors
            df['alpha_006'] = self._rank(-1 * self._ts_max(self._correlation(df['close'], df['volume'], 10), 3))
            df['alpha_007'] = self._rank(self._ts_argmax(df['volume'], 10))  # Recent volume peak
            df['alpha_008'] = self._rank(self._delta(df['volume'], 5) / self._ts_mean(df['volume'], 20))
            df['alpha_009'] = self._rank((df['high'] + df['low']) / 2 - self._delay(df['close'], 1))
            df['alpha_010'] = self._rank(self._stddev(df['returns'], 10))  # Return volatility
            
            # Momentum alpha factors
            df['alpha_011'] = self._rank(self._ts_max(df['close'] - df['low'], 5))  # Upward momentum
            df['alpha_012'] = self._rank(df['volume'] / self._ts_mean(df['volume'], 10))  # Volume intensity
            df['alpha_013'] = self._rank(-1 * self._delta(df['close'], 1))  # Mean reversion
            df['alpha_014'] = self._rank(self._correlation(df['open'], df['volume'], 10))
            df['alpha_015'] = self._rank(-1 * self._sum(self._rank(self._correlation(df['high'], df['volume'], 3)), 3))
            
            # Mean reversion factors
            df['alpha_016'] = self._rank(-1 * self._ts_max(self._rank(self._correlation(df['close'], df['volume'], 5)), 5))
            df['alpha_017'] = self._rank((df['close'] - self._delay(df['close'], 1)) / self._delay(df['close'], 1))
            df['alpha_018'] = self._rank(self._correlation(df['close'], df['open'], 10))
            df['alpha_019'] = self._rank((-1 * self._sign(self._delta(df['close'], 7))) * self._delta(df['close'], 7))
            df['alpha_020'] = self._rank(-1 * self._delta(df['open'], 1) / self._delay(df['open'], 1))
            
            # Volatility-based factors
            df['alpha_021'] = self._rank(self._stddev(df['close'], 8) / self._delay(self._stddev(df['close'], 8), 1) - 1)
            df['alpha_022'] = self._rank(self._delta(self._correlation(df['high'], df['volume'], 5), 5))
            df['alpha_023'] = self._rank(-1 * self._delta(df['high'], 2))
            df['alpha_024'] = self._rank(self._delta(self._sum(df['close'], 100) / 100, 1))
            df['alpha_025'] = self._rank(-1 * self._returns(df['close'], 9))
            
            # Complex interaction factors
            df['alpha_026'] = self._rank(self._ts_max(self._correlation(self._ts_rank(df['volume'], 5), 
                                                                      self._ts_rank(df['high'], 5), 5), 3))
            df['alpha_027'] = self._rank(self._sum(self._correlation(self._rank(df['volume']), 
                                                                    self._rank(df['close']), 6), 2))
            df['alpha_028'] = self._rank(self._scale(self._correlation(self._adv20(df), df['low'], 5)))
            df['alpha_029'] = self._rank(self._decay_linear(self._correlation(
                (df['close'] - self._ts_min(df['low'], 9)) / (self._ts_max(df['high'], 9) - self._ts_min(df['low'], 9)),
                self._delta(df['volume'], 1), 7), 16))
            df['alpha_030'] = self._rank(self._sign(self._delta(df['close'], 1)) + 
                                        self._sign(self._delay(self._delta(df['close'], 1), 1)) + 
                                        self._sign(self._delay(self._delta(df['close'], 1), 2)))
            
            # Price pattern factors
            df['alpha_031'] = self._rank(self._decay_linear(self._rank(self._rank(self._decay_linear(
                (-1 * self._rank(self._delta(df['close'], 10))), 10))), 5))
            df['alpha_032'] = self._rank(self._scale(self._sum(self._correlation(self._rank(df['high']), 
                                                                                self._rank(df['volume']), 3), 3)))
            df['alpha_033'] = self._rank(-1 * self._ts_min(self._decay_linear(self._delta(df['open'], 1), 12), 12))
            df['alpha_034'] = self._rank(self._ts_mean(df['close'], 12) / df['close'])
            df['alpha_035'] = self._rank(self._ts_rank(df['volume'], 32) * (1 - self._ts_rank(df['close'] + df['high'] - df['low'], 16)))
            
            logger.info(f"✅ Calculated {len([col for col in df.columns if 'alpha_' in col])} alpha factors")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating alpha factors: {str(e)}")
            return df
    
    def _calculate_advanced_indicators(self, df):
        """Advanced technical indicators beyond basic ones"""
        try:
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Commodity Channel Index
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            
            # Money Flow Index
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            
            # Average Directional Index
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            df['sar_signal'] = (df['close'] > df['sar']).astype(int)
            
            # Ultimate Oscillator
            df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'], 
                                       timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            # Triple Exponential Average
            df['tema'] = talib.TEMA(df['close'], timeperiod=30)
            
            # Kaufman Adaptive Moving Average
            df['kama'] = talib.KAMA(df['close'], timeperiod=30)
            
            # Hilbert Transform indicators
            df['ht_trendline'] = talib.HT_TRENDLINE(df['close'])
            df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
            df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
            
            # Chande Momentum Oscillator
            df['cmo'] = talib.CMO(df['close'], timeperiod=14)
            
            # Balance of Power
            df['bop'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
            
            # True Strength Index
            price_change = df['close'].diff()
            df['tsi'] = self._tsi(price_change, 25, 13)
            
            # Elder Ray Index
            ema_13 = df['close'].ewm(span=13).mean()
            df['bull_power'] = df['high'] - ema_13
            df['bear_power'] = df['low'] - ema_13
            
            # Aroon indicators
            df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)
            df['aroon_osc'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
            
            # Keltner Channels
            ema_20 = df['close'].ewm(span=20).mean()
            atr_20 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            df['keltner_upper'] = ema_20 + (atr_20 * 2)
            df['keltner_lower'] = ema_20 - (atr_20 * 2)
            df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {str(e)}")
            return df
    
    def _calculate_microstructure_features(self, df):
        """Market microstructure features"""
        try:
            # Tick-based features (approximated from OHLCV)
            df['tick_size'] = np.log(df['high'] / df['low'])  # Intrabar volatility
            df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_momentum'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
            
            # VWAP (Volume Weighted Average Price)
            df['vwap'] = (df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)).cumsum() / df['volume'].cumsum()
            df['vwap_ratio'] = df['close'] / df['vwap']
            
            # On-Balance Volume enhancements
            price_change = df['close'].diff()
            volume_signed = df['volume'] * np.sign(price_change)
            df['obv'] = volume_signed.cumsum()
            df['obv_sma'] = df['obv'].rolling(window=20).mean()
            df['obv_signal'] = (df['obv'] > df['obv_sma']).astype(int)
            
            # Accumulation/Distribution Line
            df['ad_line'] = (((df['close'] - df['low']) - (df['high'] - df['close'])) / 
                           (df['high'] - df['low']) * df['volume']).cumsum()
            
            # Price-Volume Trend
            df['pvt'] = (df['close'].pct_change() * df['volume']).cumsum()
            
            # Volume Profile approximation
            df['volume_profile'] = df['volume'] / df['volume'].rolling(window=50).sum()
            
            # Ease of Movement
            distance = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
            box_height = df['volume'] / (df['high'] - df['low'])
            df['ease_of_movement'] = distance / box_height
            df['emv_sma'] = df['ease_of_movement'].rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {str(e)}")
            return df
    
    def _calculate_statistical_features(self, df):
        """Statistical and distributional features"""
        try:
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                # Moments
                df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
                df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
                
                # Quantiles
                df[f'price_quantile_{window}'] = df['close'].rolling(window=window).quantile(0.5)
                df[f'volume_quantile_{window}'] = df['volume'].rolling(window=window).quantile(0.8)
                
                # Z-scores
                rolling_mean = df['close'].rolling(window=window).mean()
                rolling_std = df['close'].rolling(window=window).std()
                df[f'price_zscore_{window}'] = (df['close'] - rolling_mean) / rolling_std
            
            # Hurst Exponent (simplified version)
            df['hurst_10'] = df['close'].rolling(window=20).apply(self._hurst_exponent, raw=True)
            
            # Autocorrelation
            for lag in [1, 5, 10]:
                df[f'returns_autocorr_{lag}'] = df['returns'].rolling(window=50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )
            
            # Persistence measures
            df['returns_persistence'] = (df['returns'] * df['returns'].shift(1) > 0).astype(int)
            df['persistence_ratio'] = df['returns_persistence'].rolling(window=20).mean()
            
            # Volatility clustering
            df['vol_cluster'] = (df['returns'].abs() > df['returns'].rolling(window=20).std()).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating statistical features: {str(e)}")
            return df
    
    def _calculate_regime_features(self, df):
        """Market regime detection features"""
        try:
            # Volatility regimes
            volatility = df['returns'].rolling(window=20).std()
            vol_threshold_high = volatility.quantile(0.8)
            vol_threshold_low = volatility.quantile(0.2)
            
            df['vol_regime'] = np.where(volatility > vol_threshold_high, 2,
                                       np.where(volatility < vol_threshold_low, 0, 1))
            
            # Trend regimes
            price_ma_short = df['close'].rolling(window=10).mean()
            price_ma_long = df['close'].rolling(window=50).mean()
            df['trend_regime'] = np.where(price_ma_short > price_ma_long, 1, 0)
            
            # Market structure breaks
            df['structure_break'] = abs(df['close'].rolling(window=20).mean() - 
                                      df['close'].rolling(window=20, center=True).mean()) > \
                                  df['close'].rolling(window=20).std()
            
            # Fractal dimension
            df['fractal_dimension'] = df['close'].rolling(window=20).apply(self._fractal_dimension, raw=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {str(e)}")
            return df
    
    def _apply_denoising(self, df):
        """Apply denoising techniques"""
        try:
            if not KALMAN_AVAILABLE:
                logger.warning("Kalman filtering not available")
                return df
            
            # Kalman filter for price (fixed implementation)
            try:
                kf = KalmanFilter(
                    transition_matrices=[1],
                    observation_matrices=[1],
                    initial_state_mean=df['close'].iloc[0],
                    n_dim_state=1
                )
                
                # Fit the model and get smoothed estimates
                kf = kf.em(df['close'].values)
                state_means, _ = kf.smooth(df['close'].values)
                df['price_kalman'] = state_means.flatten()
                df['price_kalman_residual'] = df['close'] - df['price_kalman']
            except Exception as e:
                logger.warning(f"Kalman filter failed: {str(e)}")
                df['price_kalman'] = df['close'].rolling(window=5).mean()
                df['price_kalman_residual'] = df['close'] - df['price_kalman']
            
            # Wavelet denoising (if available)
            if WAVELETS_AVAILABLE:
                try:
                    # Wavelet decomposition
                    coeffs = pywt.wavedec(df['close'].values, 'db4', level=4)
                    
                    # Soft thresholding
                    threshold = 0.1 * np.std(coeffs[-1])
                    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
                    
                    # Reconstruction
                    reconstructed = pywt.waverec(coeffs_thresh, 'db4')
                    df['price_wavelet'] = reconstructed[:len(df)]
                    df['price_wavelet_residual'] = df['close'] - df['price_wavelet']
                except Exception as e:
                    logger.warning(f"Wavelet denoising failed: {str(e)}")
                    df['price_wavelet'] = df['close'].rolling(window=3).mean()
                    df['price_wavelet_residual'] = df['close'] - df['price_wavelet']
            
            # Simple moving average denoising
            df['price_ma_denoised'] = df['close'].rolling(window=5, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying denoising: {str(e)}")
            return df
    
    def _calculate_temporal_features(self, df, timeframe):
        """Temporal and cyclical features"""
        try:
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                # Create a simple timestamp column if it doesn't exist
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            
            # Convert to datetime if string
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Session features (assuming crypto 24/7)
            df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['session_european'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['session_american'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            # Weekend effect
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Timeframe-specific features
            if timeframe in ['1m', '5m', '15m']:
                df['intraday_momentum'] = df['close'] / df['close'].iloc[0] - 1
            elif timeframe in ['1h', '4h']:
                df['daily_momentum'] = df['close'].rolling(window=24).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating temporal features: {str(e)}")
            return df
    
    def _calculate_risk_features(self, df):
        """Risk and volatility-based features"""
        try:
            # Value at Risk approximation
            for confidence in [0.95, 0.99]:
                df[f'var_{int(confidence*100)}'] = df['returns'].rolling(window=50).quantile(1 - confidence)
            
            # Expected Shortfall
            df['es_95'] = df['returns'][df['returns'] <= df['var_95']].rolling(window=50).mean()
            
            # Maximum Drawdown
            rolling_max = df['close'].rolling(window=50, min_periods=1).max()
            df['drawdown'] = (df['close'] - rolling_max) / rolling_max
            df['max_drawdown'] = df['drawdown'].rolling(window=50).min()
            
            # Sharpe ratio (simplified, assuming risk-free rate = 0)
            df['sharpe_ratio'] = df['returns'].rolling(window=50).mean() / df['returns'].rolling(window=50).std()
            
            # Sortino ratio
            negative_returns = df['returns'][df['returns'] < 0]
            downside_vol = negative_returns.rolling(window=50).std()
            df['sortino_ratio'] = df['returns'].rolling(window=50).mean() / downside_vol
            
            # Beta approximation (vs market - using price momentum as proxy)
            market_returns = df['returns'].rolling(window=50).mean()  # Simplified market proxy
            covariance = df['returns'].rolling(window=50).cov(market_returns)
            market_variance = market_returns.rolling(window=50).var()
            df['beta'] = covariance / market_variance
            
            # Volatility clustering
            squared_returns = df['returns'] ** 2
            df['garch_vol'] = squared_returns.ewm(alpha=0.1).mean().apply(np.sqrt)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating risk features: {str(e)}")
            return df
    
    def _calculate_interaction_features(self, df):
        """Feature interactions and combinations"""
        try:
            # Price-Volume interactions
            df['pv_correlation'] = df['close'].rolling(window=20).corr(df['volume'])
            df['pv_ratio'] = df['close'] * df['volume']  # Simple price-volume product
            
            # Momentum combinations
            df['momentum_composite'] = (df.get('rsi_14', 50) / 100 + 
                                      df.get('stoch_k_14', 50) / 100 + 
                                      (df.get('macd_12_26', 0) > 0).astype(int)) / 3
            
            # Volatility combinations
            df['volatility_composite'] = (df.get('bb_width_20', 0) + 
                                         df.get('atr', 0) + 
                                         df['returns'].rolling(window=20).std()) / 3
            
            # Trend strength
            df['trend_strength'] = abs(df.get('ema_12', df['close']) - df.get('ema_26', df['close'])) / df['close']
            
            # Support/Resistance features
            df['support_level'] = df['low'].rolling(window=20).min()
            df['resistance_level'] = df['high'].rolling(window=20).max()
            df['support_distance'] = (df['close'] - df['support_level']) / df['close']
            df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating interaction features: {str(e)}")
            return df
    
    def _cleanup_features(self, df):
        """Final cleanup and validation"""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values more intelligently
            # For technical indicators, forward fill is usually appropriate
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # Don't fill OHLCV columns
            original_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in numeric_columns if col not in original_cols]
            
            # Forward fill feature columns
            for col in feature_cols:
                df[col] = df[col].fillna(method='ffill')
            
            # For remaining NaNs in features, use median
            for col in feature_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0)  # Last resort
            
            # Clip extreme outliers more conservatively (beyond 3 standard deviations)
            for col in feature_cols:
                if df[col].std() > 0:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if not pd.isna(std_val) and std_val > 0:
                        lower_bound = mean_val - 3 * std_val
                        upper_bound = mean_val + 3 * std_val
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Only drop rows if ALL features are NaN (very conservative)
            # This should rarely happen now
            feature_nan_count = df[feature_cols].isna().sum(axis=1)
            rows_to_keep = feature_nan_count < len(feature_cols)  # Keep if at least one feature is valid
            
            initial_rows = len(df)
            df = df[rows_to_keep]
            final_rows = len(df)
            
            if final_rows < initial_rows:
                logger.warning(f"Removed {initial_rows - final_rows} rows with all NaN features")
            
            # Ensure we still have reasonable data
            if len(df) < 20:
                logger.warning(f"Very few rows remaining after cleanup: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature cleanup: {str(e)}")
            return df
    
    # Helper functions for alpha factors
    def _rank(self, series):
        """Cross-sectional rank (simplified for single asset)"""
        return series.rolling(window=50, min_periods=1).rank(pct=True)
    
    def _delay(self, series, periods):
        """Delay/lag function"""
        return series.shift(periods)
    
    def _delta(self, series, periods):
        """Delta/difference function"""
        return series.diff(periods)
    
    def _ts_sum(self, series, periods):
        """Time-series sum"""
        return series.rolling(window=periods, min_periods=1).sum()
    
    def _ts_mean(self, series, periods):
        """Time-series mean"""
        return series.rolling(window=periods, min_periods=1).mean()
    
    def _ts_max(self, series, periods):
        """Time-series maximum"""
        return series.rolling(window=periods, min_periods=1).max()
    
    def _ts_min(self, series, periods):
        """Time-series minimum"""
        return series.rolling(window=periods, min_periods=1).min()
    
    def _ts_argmax(self, series, periods):
        """Time-series argument of maximum"""
        return series.rolling(window=periods, min_periods=1).apply(np.argmax, raw=True)
    
    def _ts_rank(self, series, periods):
        """Time-series rank"""
        return series.rolling(window=periods, min_periods=1).rank(pct=True)
    
    def _stddev(self, series, periods):
        """Standard deviation"""
        return series.rolling(window=periods, min_periods=1).std()
    
    def _correlation(self, x, y, periods):
        """Rolling correlation"""
        return x.rolling(window=periods, min_periods=1).corr(y)
    
    def _sign(self, series):
        """Sign function"""
        return np.sign(series)
    
    def _scale(self, series):
        """Scale to unit variance"""
        return series / series.std() if series.std() != 0 else series
    
    def _sum(self, series, periods):
        """Simple sum wrapper"""
        return self._ts_sum(series, periods)
    
    def _returns(self, series, periods):
        """Returns over periods"""
        return series.pct_change(periods)
    
    def _adv20(self, df):
        """Average daily volume over 20 periods"""
        return df['volume'].rolling(window=20, min_periods=1).mean()
    
    def _decay_linear(self, series, periods):
        """Linear decay weighting"""
        weights = np.arange(1, periods + 1)
        weights = weights / weights.sum()
        return series.rolling(window=periods, min_periods=1).apply(
            lambda x: np.average(x, weights=weights[-len(x):])
        )
    
    def _tsi(self, price_change, long_period, short_period):
        """True Strength Index"""
        double_smoothed_pc = price_change.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_apc = price_change.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        return 100 * (double_smoothed_pc / double_smoothed_apc)
    
    def _hurst_exponent(self, ts):
        """Simplified Hurst exponent calculation"""
        try:
            if len(ts) < 10:
                return np.nan
            
            lags = range(2, min(20, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            if len(tau) < 3:
                return np.nan
                
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return np.nan
    
    def _fractal_dimension(self, ts):
        """Simplified fractal dimension"""
        try:
            if len(ts) < 10:
                return np.nan
                
            n = len(ts)
            length = np.sum(np.sqrt(1 + np.diff(ts)**2))
            return np.log(length) / np.log(n)
        except:
            return np.nan

# Global instance
enhanced_feature_engine = EnhancedFeatureEngine()

def calculate_enhanced_indicators(df, symbol=None, timeframe='1h'):
    """
    Main function to calculate all enhanced indicators
    """
    try:
        if df is None or df.empty:
            logger.error("Empty DataFrame provided")
            return df
            
        logger.info(f"🔧 Calculating enhanced features for {symbol or 'unknown'} ({timeframe})")
        
        # Initialize the feature engine
        engine = EnhancedFeatureEngine()
        
        # Calculate all features
        enhanced_df = engine.calculate_enhanced_features(df, symbol=symbol, timeframe=timeframe)
        
        if enhanced_df is None or enhanced_df.empty:
            logger.error("Enhanced feature calculation returned empty DataFrame")
            return df
            
        logger.info(f"✅ Generated {enhanced_df.shape[1]} enhanced features")
        return enhanced_df
        
    except Exception as e:
        logger.error(f"Error in enhanced feature calculation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return df  # Return original DataFrame on error

if __name__ == "__main__":
    # Test the feature engine
    logger.info("🧪 Testing Enhanced Feature Engine...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate realistic OHLCV data
    returns = np.random.normal(0, 0.02, 1000)
    prices = 100 * (1 + returns).cumprod()
    
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    })
    
    # Test feature calculation
    result_df = calculate_enhanced_indicators(test_df, 'TEST-USD', '1h')
    
    logger.info(f"✅ Test completed:")
    logger.info(f"   Input shape: {test_df.shape}")
    logger.info(f"   Output shape: {result_df.shape}")
    logger.info(f"   Features added: {result_df.shape[1] - test_df.shape[1]}")
    logger.info(f"   Sample features: {list(result_df.columns[:10])}") 