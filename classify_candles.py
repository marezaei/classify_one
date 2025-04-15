"""
Candle Pattern Classification Script
------------------------------------
This script loads SPY one-minute bar data from a pickle file, calculates candle
features, classifies the candles into 9 classes using gradient boosting,
saves the classified data, and creates visualizations.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
from datetime import datetime
import mplfinance as mpf
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_latest_pickle():
    """Load the most recent pickle file from the data directory."""
    logger.info("Looking for latest pickle file in data directory...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    pickle_files = [f for f in os.listdir(data_dir) if f.startswith('spy_one_min_data_') and f.endswith('.pkl')]
    
    if not pickle_files:
        logger.error("No pickle files found. Run export_to_pickle.py first.")
        raise FileNotFoundError("No pickle files found. Run export_to_pickle.py first.")
    
    # Sort by name (which includes date) to get the latest
    latest_file = sorted(pickle_files)[-1]
    file_path = os.path.join(data_dir, latest_file)
    
    logger.info(f"Loading data from: {file_path}")
    df = pd.read_pickle(file_path)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    return df

def calculate_candle_features(df):
    """Calculate features for candle pattern classification."""
    logger.info("Calculating candle features...")
    start_time = time.time()
    
    # Calculate basic features
    logger.info("Computing basic size metrics...")
    df['body_size'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Calculate ratios (with handling for division by zero)
    logger.info("Computing ratio metrics...")
    # Body to Range ratio
    df['body_to_range_ratio'] = np.where(
        df['total_range'] > 0,
        df['body_size'] / df['total_range'],
        0
    )
    
    # Upper Shadow to Body ratio (if body size > 0)
    df['upper_shadow_to_body_ratio'] = np.where(
        df['body_size'] > 0,
        df['upper_shadow'] / df['body_size'],
        0
    )
    
    # Lower Shadow to Body ratio (if body size > 0)
    df['lower_shadow_to_body_ratio'] = np.where(
        df['body_size'] > 0,
        df['lower_shadow'] / df['body_size'],
        0
    )
    
    # Body Position
    df['body_position'] = np.where(
        df['total_range'] > 0,
        ((df['open'] + df['close']) / 2 - df['low']) / df['total_range'],
        0.5  # Default to middle if no range
    )
    
    # Bullish or Bearish
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    logger.info(f"Candle feature calculation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Feature statistics:\n{df[['body_size', 'total_range', 'upper_shadow', 'lower_shadow']].describe()}")
    return df

def classify_candles(df):
    """Classify candles into 9 distinct classes using K-means clustering first,
    then train a Gradient Boosting model to predict these classes."""
    logger.info("Starting candle classification process...")
    
    # Features for classification
    feature_cols = [
        'body_size', 'total_range', 'upper_shadow', 'lower_shadow',
        'body_to_range_ratio', 'upper_shadow_to_body_ratio',
        'lower_shadow_to_body_ratio'
    ]
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    logger.info(f"Feature scaling complete. Mean: {scaler.mean_}, Variance: {scaler.var_}")
    
    # Use K-means to create initial labels (unsupervised)
    logger.info("Running K-means clustering to create initial labels...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=9, random_state=42, verbose=1)
    df['kmeans_class'] = kmeans.fit_predict(scaled_features)
    logger.info(f"K-means clustering completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"K-means cluster centers:\n{kmeans.cluster_centers_}")
    logger.info(f"K-means cluster distribution: {pd.Series(df['kmeans_class']).value_counts().to_dict()}")
    
    # Train a Gradient Boosting model on these labels
    logger.info("Splitting data for Gradient Boosting training...")
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, df['kmeans_class'], test_size=0.2, random_state=42
    )
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    logger.info("Training Gradient Boosting model...")
    start_time = time.time()
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=1
    )
    gb_model.fit(X_train, y_train)
    logger.info(f"Gradient Boosting model training completed in {time.time() - start_time:.2f} seconds")
    
    # Predict classes using the Gradient Boosting model
    logger.info("Predicting candle classes...")
    df['candle_class'] = gb_model.predict(scaled_features)
    
    # Calculate accuracy on test set
    accuracy = gb_model.score(X_test, y_test)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    logger.info(f"Class distribution: {pd.Series(df['candle_class']).value_counts().to_dict()}")
    
    # Save the model and scaler for future use
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    model_path = os.path.join(output_dir, f'candle_classifier_model_{timestamp}.pkl')
    scaler_path = os.path.join(output_dir, f'candle_feature_scaler_{timestamp}.pkl')
    
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(gb_model, f)
    
    logger.info(f"Saving scaler to {scaler_path}")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Model and scaler saved to {output_dir}")
    
    return df, gb_model, scaler

def save_classified_data(df):
    """Save the classified candle data to a pickle file."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f'classified_candles_{timestamp}.pkl')
    
    logger.info(f"Saving classified data with shape {df.shape} to {output_path}")
    df.to_pickle(output_path)
    logger.info(f"Classified candle data saved successfully")
    return output_path

def create_visualizations(df, model):
    """Create visualizations for the candle classifications."""
    logger.info("Creating visualizations...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature importance plot
    logger.info("Creating feature importance plot...")
    feature_cols = [
        'body_size', 'total_range', 'upper_shadow', 'lower_shadow',
        'body_to_range_ratio', 'upper_shadow_to_body_ratio',
        'lower_shadow_to_body_ratio', 'body_position'
    ]
    importances = model.feature_importances_
    logger.info(f"Feature importances: {dict(zip(feature_cols, importances))}")
    
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance for Candle Classification')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    logger.info("Feature importance plot saved")
    
    # 2. Candle class distribution
    logger.info("Creating class distribution plot...")
    plt.figure(figsize=(12, 6))
    class_counts = df['candle_class'].value_counts()
    logger.info(f"Class counts:\n{class_counts}")
    sns.countplot(x='candle_class', data=df)
    plt.title('Distribution of Candle Classes')
    plt.xlabel('Candle Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    logger.info("Class distribution plot saved")
    
    # 3. Sample candles from each class
    logger.info("Creating sample candle plots for each class...")
    for class_num in range(9):
        logger.info(f"Processing class {class_num}...")
        class_samples = df[df['candle_class'] == class_num].head(5)
        logger.info(f"Found {len(class_samples)} samples for class {class_num}")
        if len(class_samples) > 0:
            # Format data for mplfinance
            # Explicitly name the reset index column to avoid KeyError
            class_samples_plot = class_samples.reset_index(names=['Date'])
            class_samples_plot = class_samples_plot.rename(
                columns={'open': 'Open', 'high': 'High', 
                         'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            )
            class_samples_plot.set_index('Date', inplace=True)
            
            # Log column names for debugging
            logger.info(f"DataFrame columns after reset_index: {class_samples_plot.columns.tolist()}")
            
            # Plot candles
            fig, axs = plt.subplots(1, len(class_samples), figsize=(15, 5))
            for i, (idx, row) in enumerate(class_samples.iterrows()):
                if len(class_samples) > 1:
                    ax = axs[i]
                else:
                    ax = axs
                
                # Create a single-candle dataframe
                candle_df = pd.DataFrame({
                    'Open': [row['open']],
                    'High': [row['high']],
                    'Low': [row['low']],
                    'Close': [row['close']],
                    'Volume': [row['volume']]
                }, index=[idx])
                
                # Remove the invalid xaxis parameter and hide x-axis after plotting
                mpf.plot(candle_df, type='candle', ax=ax, style='yahoo')
                ax.set_xticklabels([])  # Hide x-axis labels
                ax.set_xlabel('')        # Remove x-axis label
                ax.set_title(f"Sample {i+1}")
            
            plt.suptitle(f"Class {class_num} Candle Samples")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'class_{class_num}_samples.png'))
            plt.close()
            logger.info(f"Saved sample plot for class {class_num}")
    
    # 4. Feature distributions by class
    logger.info("Creating feature distribution plots...")
    for feature in feature_cols:
        logger.info(f"Creating boxplot for feature: {feature}")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='candle_class', y=feature, data=df)
        plt.title(f'{feature} Distribution by Candle Class')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_by_class.png'))
        plt.close()
    
    logger.info("Creating class overlap visualizations...")
    
    # Sample data to make visualization more manageable
    if len(df) > 10000:
        logger.info(f"Sampling data for dimensionality reduction (full size: {len(df)})")
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df
    
    features_sample = df_sample[feature_cols]
    
    # Standardize the features for better dimensionality reduction
    logger.info("Standardizing features for dimensionality reduction...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_sample)
    
    # PCA to 2 dimensions
    logger.info("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=df_sample['candle_class'], cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Candle Class')
    plt.title('PCA - Candle Classes Visualization')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_overlap_pca.png'))
    plt.close()
    logger.info("PCA visualization saved")
    
    # t-SNE visualization (more expensive but often better for cluster visualization)
    logger.info("Performing t-SNE dimensionality reduction (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_result = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                         c=df_sample['candle_class'], cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Candle Class')
    plt.title('t-SNE - Candle Classes Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_overlap_tsne.png'))
    plt.close()
    logger.info("t-SNE visualization saved")
    
    # 6. Pairplot of the most important features
    logger.info("Creating pairplot of top features...")
    # Get top 4 features based on importance
    indices = np.argsort(model.feature_importances_)[::-1]
    top_features = [feature_cols[i] for i in indices[:4]]
    
    # Create a subset dataframe with top features and class
    top_features_df = df_sample[top_features + ['candle_class']].copy()
    
    # Create pairplot
    pairplot = sns.pairplot(top_features_df, hue='candle_class', palette='viridis', 
                            plot_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'k'}, 
                            corner=True)
    plt.suptitle('Pairplot of Top Features by Class', y=1.02)
    pairplot.savefig(os.path.join(output_dir, 'top_features_pairplot.png'))
    plt.close()
    logger.info("Top features pairplot saved")
    
    # 7. 3D scatter plot of top 3 features
    if len(top_features) >= 3:
        logger.info("Creating 3D plot of top 3 features...")
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            top_features_df[top_features[0]], 
            top_features_df[top_features[1]], 
            top_features_df[top_features[2]],
            c=top_features_df['candle_class'],
            cmap='viridis',
            s=40,
            alpha=0.6
        )
        
        ax.set_xlabel(top_features[0])
        ax.set_ylabel(top_features[1])
        ax.set_zlabel(top_features[2])
        ax.set_title('3D Plot of Top 3 Features by Class')
        
        # Add color bar
        fig.colorbar(scatter, ax=ax, label='Candle Class')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_features_3d.png'))
        plt.close()
        logger.info("3D plot of top features saved")
    
    # 8. Class correlation matrix - how features relate to classes
    logger.info("Creating class feature correlation heatmap...")
    # Create dummy variables for class labels
    class_dummies = pd.get_dummies(df_sample['candle_class'], prefix='class')
    # Combine with features
    correlation_df = pd.concat([df_sample[feature_cols], class_dummies], axis=1)
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Get only the correlations between features and classes
    class_cols = [col for col in corr_matrix.columns if col.startswith('class_')]
    feature_class_corr = corr_matrix.loc[feature_cols, class_cols]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(feature_class_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature-Class Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_class_correlation.png'))
    plt.close()
    logger.info("Feature-class correlation heatmap saved")
    
    logger.info(f"All visualizations saved to {output_dir}")

def main():
    logger.info("Starting candle pattern classification process...")
    start_time = time.time()
    
    # Load data
    df = load_latest_pickle()
    
    # Calculate candle features
    df = calculate_candle_features(df)
    
    # Classify candles
    df, model, scaler = classify_candles(df)
    
    # Save classified data
    save_classified_data(df)
    
    # Create visualizations
    create_visualizations(df, model)
    
    logger.info(f"Candle classification complete! Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
