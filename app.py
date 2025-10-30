import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import ollama 

md = "gemma3:1b"

# Set page configuration
st.set_page_config(
    page_title="DQNN-TorchTrack Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_experiment_data():
    """Load experiment data from data.json."""
    try:
        if not os.path.exists('data.json'):
            return pd.DataFrame(), []
        
        with open('data.json', 'r') as f:
            data = json.load(f)
        
        if not data:
            return pd.DataFrame(), []
        
        # Convert to DataFrame for easy manipulation
        df_records = []
        for exp in data:
            record = {**exp.get('hyperparameters', {}), 
                     **exp.get('metrics', {}), 
                     'experiment_name': exp.get('experiment_name', 'Unknown')}
            df_records.append(record)
        
        df = pd.DataFrame(df_records)
        return df, data
        
    except Exception as e:
        st.error(f"Error loading experiment data: {e}")
        return pd.DataFrame(), []

def load_best_architecture():
    """Load the best architecture found by DQNN."""
    try:
        if not os.path.exists('best_architecture.json'):
            return None
            
        with open('best_architecture.json', 'r') as f:
            best_arch = json.load(f)
        return best_arch
    except Exception as e:
        st.error(f"Error loading best architecture: {e}")
        return None

def plot_architecture_evolution(df):
    """Plot how architecture performance evolved over time."""
    if df.empty:
        st.warning("No data available for architecture evolution plot.")
        return None
    
    # Sort by experiment name to get chronological order
    df_sorted = df.sort_values('experiment_name')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Over Time', 'Model Complexity', 
                       'Training Time', 'Architecture Size'),
        vertical_spacing=0.12
    )
    
    # Performance over time
    performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
    if performance_col in df.columns:
        fig.add_trace(
            go.Scatter(x=list(range(len(df_sorted))), y=df_sorted[performance_col], 
                      mode='lines+markers', name='Performance',
                      line=dict(color='blue')),
            row=1, col=1
        )
    
    # Model complexity
    if 'model_complexity' in df.columns:
        fig.add_trace(
            go.Scatter(x=list(range(len(df_sorted))), y=df_sorted['model_complexity'], 
                      mode='lines+markers', name='Model Complexity',
                      line=dict(color='red')),
            row=1, col=2
        )
    
    # Training time
    if 'training_time' in df.columns:
        fig.add_trace(
            go.Scatter(x=list(range(len(df_sorted))), y=df_sorted['training_time'], 
                      mode='lines+markers', name='Training Time',
                      line=dict(color='green')),
            row=2, col=1
        )
    
    # Architecture diversity (estimate based on layers if available)
    try:
        if 'layers' in df.columns:
            layer_counts = []
            for layers in df_sorted['layers']:
                if isinstance(layers, list):
                    layer_counts.append(len(layers))
                else:
                    layer_counts.append(1)
            
            fig.add_trace(
                go.Scatter(x=list(range(len(df_sorted))), y=layer_counts, 
                          mode='lines+markers', name='Number of Layers',
                          line=dict(color='purple')),
                row=2, col=2
            )
    except Exception as e:
        st.warning(f"Could not plot architecture diversity: {e}")
    
    fig.update_layout(height=600, title_text="DQNN Architecture Search Evolution", showlegend=False)
    return fig

def plot_hyperparameter_analysis(df):
    """Create comprehensive hyperparameter analysis plots."""
    if df.empty:
        st.warning("No data available for hyperparameter analysis.")
        return None
    
    # Prepare data for analysis
    analysis_data = []
    performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
    
    for idx, row in df.iterrows():
        try:
            entry = {
                'experiment': row.get('experiment_name', f'Exp_{idx}'),
                'learning_rate': float(row.get('learning_rate', 0.01)),
                'batch_size': int(row.get('batch_size', 32)),
                'epochs': int(row.get('epochs', 100)),
                'performance': float(row.get(performance_col, 0))
            }
            
            # Handle layers data
            layers = row.get('layers', [64])
            if isinstance(layers, list):
                entry['num_layers'] = len(layers)
                entry['total_neurons'] = sum(layers)
                entry['avg_layer_size'] = np.mean(layers)
            else:
                entry['num_layers'] = 1
                entry['total_neurons'] = 64
                entry['avg_layer_size'] = 64
            
            analysis_data.append(entry)
        except Exception as e:
            continue
    
    if not analysis_data:
        st.warning("Could not process data for hyperparameter analysis.")
        return None
    
    analysis_df = pd.DataFrame(analysis_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Layers vs Performance', 'Learning Rate vs Performance',
                       'Network Size vs Performance', 'Batch Size vs Performance'),
        vertical_spacing=0.12
    )
    
    # Number of layers vs performance
    fig.add_trace(
        go.Scatter(x=analysis_df['num_layers'], y=analysis_df['performance'],
                  mode='markers', name='Layers vs Perf',
                  marker=dict(size=8, color='blue')),
        row=1, col=1
    )
    
    # Learning rate vs performance
    fig.add_trace(
        go.Scatter(x=analysis_df['learning_rate'], y=analysis_df['performance'],
                  mode='markers', name='LR vs Perf',
                  marker=dict(size=8, color='red')),
        row=1, col=2
    )
    
    # Total neurons vs performance
    fig.add_trace(
        go.Scatter(x=analysis_df['total_neurons'], y=analysis_df['performance'],
                  mode='markers', name='Size vs Perf',
                  marker=dict(size=8, color='green')),
        row=2, col=1
    )
    
    # Batch size vs performance
    fig.add_trace(
        go.Scatter(x=analysis_df['batch_size'], y=analysis_df['performance'],
                  mode='markers', name='Batch vs Perf',
                  marker=dict(size=8, color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Hyperparameter Impact Analysis", showlegend=False)
    return fig

def visualize_best_architecture(best_arch):
    """Create a visual representation of the best architecture."""
    if not best_arch or 'layers' not in best_arch:
        st.warning("No architecture data available for visualization.")
        return None
    
    layers = best_arch['layers']
    activations = best_arch.get('activations', ['relu'] * len(layers))
    
    # Create architecture diagram using text-based visualization
    st.subheader("Architecture Structure")
    
    # Display as a flow diagram
    cols = st.columns(len(layers) + 2)  # +2 for input and output
    
    with cols[0]:
        st.markdown("**Input**")
        st.markdown("Layer")
    
    for i, (size, activation) in enumerate(zip(layers, activations)):
        with cols[i + 1]:
            st.markdown(f"**Hidden {i+1}**")
            st.markdown(f"{size} neurons")
            st.markdown(f"{activation}")
    
    with cols[-1]:
        st.markdown("**Output**")
        st.markdown("Layer")
    
    # Create a simple bar chart showing layer sizes
    fig = go.Figure(data=[
        go.Bar(x=[f'Layer {i+1}' for i in range(len(layers))], 
               y=layers,
               text=[f'{size}<br>{act}' for size, act in zip(layers, activations)],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Layer Sizes and Activations",
        xaxis_title="Layers",
        yaxis_title="Number of Neurons",
        height=400
    )
    
    return fig

def main():
    st.title('DQNN-TorchTrack: Neural Architecture Search Dashboard')
    st.markdown("---")
    
    # Load data
    df, full_data = load_experiment_data()
    best_arch = load_best_architecture()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a view", [
        "Overview", "Best Architecture", "Performance Analysis", 
        "Hyperparameter Study", "Detailed Results", "Export Results",
        "AI insights"
    ])
    
    if page == "Overview":
        st.header("Experiment Overview")
        
        if df.empty:
            st.warning("No experiment data available. Run the DQNN training first!")
            st.info("To get started, run your DQNN training script to generate experiment data.")
            return
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Display task type information
        if best_arch and 'task_type' in best_arch:
            task_type = best_arch['task_type']
            st.info(f"🎯 **Task Type:** {task_type.replace('_', ' ').title()}")
            
            if 'num_classes' in best_arch and best_arch['num_classes']:
                st.info(f"📊 **Number of Classes:** {best_arch['num_classes']}")

        with col1:
            st.metric("Total Experiments", len(df))
        
        with col2:
            performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
            if performance_col in df.columns:
                best_perf = df[performance_col].max()
                st.metric("Best Performance", f"{best_perf:.4f}")
        
        with col3:
            if 'training_time' in df.columns:
                avg_time = df['training_time'].mean()
                st.metric("Avg Training Time", f"{avg_time:.2f}s")
        
        with col4:
            if best_arch and 'layers' in best_arch:
                st.metric("Best Architecture Layers", len(best_arch['layers']))
        
        
        
        # Evolution plot
        st.subheader("Architecture Search Evolution")
        evolution_fig = plot_architecture_evolution(df)
        if evolution_fig:
            st.plotly_chart(evolution_fig, use_container_width=True)
        
        # Quick stats
        st.subheader("Quick Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance distribution
            performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
            if performance_col in df.columns:
                fig_hist = px.histogram(df, x=performance_col, 
                                      title="Performance Distribution",
                                      nbins=min(20, len(df)))
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Architecture complexity distribution
            if 'model_complexity' in df.columns:
                fig_complexity = px.histogram(df, x='model_complexity',
                                             title="Model Complexity Distribution",
                                             nbins=min(20, len(df)))
                st.plotly_chart(fig_complexity, use_container_width=True)
    
    elif page == "Best Architecture":
        st.header("Best Architecture Found")
        
        if best_arch:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Architecture Details")
                st.json(best_arch)
            
            with col2:
                st.subheader("Performance Metrics")
                perf = best_arch.get('performance', 'N/A')
                if isinstance(perf, (int, float)):
                    st.metric("Performance", f"{perf:.4f}")
                else:
                    st.metric("Performance", str(perf))
                
                st.markdown("**Architecture Summary:**")
                st.write(f"- **Task Type:** {best_arch.get('task_type', 'N/A').replace('_', ' ').title()}")
                
                if 'num_classes' in best_arch and best_arch['num_classes']:
                    st.write(f"- **Number of Classes:** {best_arch['num_classes']}")
                
                st.write(f"- **Layers:** {best_arch.get('layers', 'N/A')}")
                st.write(f"- **Activations:** {best_arch.get('activations', 'N/A')}")
                st.write(f"- **Learning Rate:** {best_arch.get('learning_rate', 'N/A')}")
                st.write(f"- **Batch Size:** {best_arch.get('batch_size', 'N/A')}")
                st.write(f"- **Epochs:** {best_arch.get('epochs', 'N/A')}")
                
                if 'dropout_rate' in best_arch:
                    st.write(f"- **Dropout Rate:** {best_arch['dropout_rate']}")
            
            # Visual representation
            st.subheader("Architecture Visualization")
            arch_fig = visualize_best_architecture(best_arch)
            if arch_fig:
                st.plotly_chart(arch_fig, use_container_width=True)
            
            # Code generation for best architecture
            st.subheader("PyTorch Code for Best Architecture")
            
            task_type = best_arch.get('task_type', 'classification')
            output_activation = ""
            
            if task_type == 'binary_classification':
                output_activation = "        layers.append(nn.Sigmoid())  # Binary classification"
            elif task_type == 'multiclass_classification':
                output_activation = "        # No activation - use CrossEntropyLoss which includes softmax"
            else:
                output_activation = "        # No activation for regression"
            
            code = f"""
import torch
import torch.nn as nn

class BestArchitecture(nn.Module):
    def __init__(self, input_size, output_size):
        super(BestArchitecture, self).__init__()
        layers = []
        prev_size = input_size
        
        # Hidden layers
        layer_sizes = {best_arch.get('layers', [64])}
        activations = {best_arch.get('activations', ['relu'])}
        dropout_rate = {best_arch.get('dropout_rate', 0.0)}
        
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))
            
            if i < len(activations):
                if activations[i] == 'relu':
                    layers.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    layers.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    layers.append(nn.Sigmoid())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
{output_activation}
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Task type: {task_type.replace('_', ' ').title()}
# Recommended hyperparameters
learning_rate = {best_arch.get('learning_rate', 0.01)}
batch_size = {best_arch.get('batch_size', 32)}
epochs = {best_arch.get('epochs', 100)}

# Recommended loss function:
# - Binary Classification: nn.BCELoss()
# - Multiclass Classification: nn.CrossEntropyLoss()
# - Regression: nn.MSELoss()
"""
            st.code(code, language='python')
        
        else:
            st.warning("No best architecture data found. Run DQNN training first!")
    
    elif page == "Performance Analysis":
        st.header("Performance Analysis")
        
        if df.empty:
            st.warning("No data available for performance analysis.")
            return
        
        # Hyperparameter impact analysis
        st.subheader("Hyperparameter Impact on Performance")
        hyper_fig = plot_hyperparameter_analysis(df)
        if hyper_fig:
            st.plotly_chart(hyper_fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        # Prepare numerical data for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               title="Correlation Matrix of Experiment Parameters",
                               color_continuous_scale='RdBu_r',
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Performance vs different metrics
        st.subheader("Performance Relationships")
        
        performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
        if performance_col in df.columns:
            
            # Create scatter plots for key relationships
            if 'model_complexity' in df.columns and 'training_time' in df.columns:
                fig_3d = px.scatter_3d(df, 
                                     x='model_complexity', 
                                     y='training_time', 
                                     z=performance_col,
                                     title="Performance vs Complexity vs Training Time",
                                     labels={
                                         'model_complexity': 'Model Complexity',
                                         'training_time': 'Training Time (s)',
                                         performance_col: 'Performance'
                                     })
                st.plotly_chart(fig_3d, use_container_width=True)
    
    elif page == "Hyperparameter Study":
        st.header("Hyperparameter Study")
        
        if df.empty:
            st.warning("No data available for hyperparameter study.")
            return
        
        # Architecture analysis
        st.subheader("Architecture Analysis")
        
        # Extract architecture information
        arch_data = []
        performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
        
        for idx, row in df.iterrows():
            try:
                layers = row.get('layers', [64])
                if isinstance(layers, list) and layers:
                    arch_data.append({
                        'experiment': row.get('experiment_name', f'Exp_{idx}'),
                        'num_layers': len(layers),
                        'min_layer_size': min(layers),
                        'max_layer_size': max(layers),
                        'total_neurons': sum(layers),
                        'performance': float(row.get(performance_col, 0))
                    })
            except Exception:
                continue
        
        if arch_data:
            arch_df = pd.DataFrame(arch_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Layer count analysis
                layer_perf = arch_df.groupby('num_layers')['performance'].agg(['mean', 'std', 'count']).reset_index()
                fig_layers = px.bar(layer_perf, x='num_layers', y='mean',
                                  error_y='std',
                                  title="Average Performance by Number of Layers")
                st.plotly_chart(fig_layers, use_container_width=True)
            
            with col2:
                # Network size analysis
                fig_size = px.scatter(arch_df, x='total_neurons', y='performance',
                                    title="Performance vs Total Network Size",
                                    trendline="ols")
                st.plotly_chart(fig_size, use_container_width=True)
            
            # Detailed architecture table
            st.subheader("Architecture Performance Summary")
            st.dataframe(arch_df.sort_values('performance', ascending=False))
        
        # Hyperparameter ranges analysis
        st.subheader("Hyperparameter Ranges Explored")
        
        param_summary = {}
        for col in ['learning_rate', 'batch_size', 'epochs', 'dropout_rate']:
            if col in df.columns:
                param_summary[col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        if param_summary:
            summary_df = pd.DataFrame(param_summary).T
            st.dataframe(summary_df)
    
    elif page == "Detailed Results":
        st.header("Detailed Experiment Results")
        
        if df.empty:
            st.warning("No experiment data available.")
            return
        
        # Sort options
        sort_columns = ['experiment_name']
        for col in ['accuracy', 'r2_score', 'training_time', 'model_complexity']:
            if col in df.columns:
                sort_columns.append(col)
        
        sort_by = st.selectbox("Sort by:", sort_columns)
        ascending = st.checkbox("Ascending order", value=False)
        
        if sort_by in df.columns:
            df_sorted = df.sort_values(by=sort_by, ascending=ascending)
        else:
            df_sorted = df
        
        # Display detailed table
        st.subheader('All Experiment Results')
        st.dataframe(df_sorted, use_container_width=True)
        
        # Epoch-wise performance plots
        if full_data:
            st.subheader('Epoch-wise Performance Analysis')
            
            # Select experiments to compare
            experiment_names = [exp.get('experiment_name', f'Exp_{i}') for i, exp in enumerate(full_data)]
            selected_experiments = st.multiselect(
                "Select experiments to compare:",
                options=experiment_names,
                default=experiment_names[:min(3, len(experiment_names))]
            )
            
            if selected_experiments:
                fig_epochs = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Loss per Epoch', 'Performance per Epoch')
                )
                
                for exp in full_data:
                    exp_name = exp.get('experiment_name', 'Unknown')
                    if exp_name in selected_experiments:
                        epoch_data = exp.get('epoch_data', {})
                        loss_data = epoch_data.get('loss_per_epoch', [])
                        perf_data = epoch_data.get('accuracy_per_epoch', [])
                        
                        if loss_data:
                            epochs = list(range(1, len(loss_data) + 1))
                            
                            # Loss plot
                            fig_epochs.add_trace(
                                go.Scatter(x=epochs, y=loss_data,
                                         mode='lines', name=f"{exp_name} - Loss"),
                                row=1, col=1
                            )
                        
                        if perf_data:
                            epochs = list(range(1, len(perf_data) + 1))
                            
                            # Performance plot
                            fig_epochs.add_trace(
                                go.Scatter(x=epochs, y=perf_data,
                                         mode='lines', name=f"{exp_name} - Performance"),
                                row=1, col=2
                            )
                
                fig_epochs.update_layout(height=400, title_text="Training Progress Comparison")
                st.plotly_chart(fig_epochs, use_container_width=True)

    elif page == "AI insights":
        
        st.header("AI-Powered Insights")
        
        if df.empty:
            st.warning("No data available for AI insights.")
            return
        
        # Get best architecture and model type
        best_arch = load_best_architecture()
        model_type = best_arch.get('task_type', 'classification') if best_arch else 'classification'
        
        # Prepare data summary
        performance_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
        df_sorted = df.sort_values(by=performance_col, ascending=False) if performance_col in df.columns else df
        
        # Get model data from full_data
        model_data_info = ""
        if full_data:
            for exp in full_data[:3]:  # Use top 3 experiments
                model_data_info += f"\n{exp.get('model_data', 'N/A')}"

        with st.expander("Model performance analysis"):
                response = ollama.chat(
                    model=md,
                    messages=[{
                        "role": "user", 
                        "content": f"""
                        You are an expert on analyzing ml models, 
                        Generate an in detail performance analysis of different models based on the info I give you:
                        The model is a {exp["model_type"]} model with given params and results: 
                        {df_sorted.to_string()},
                        Model data:
                        {exp["model_data"]}
                        """
                    }], stream=True
                )
                res_content = ""
                def catch_response(response):
                    global response_content
                    for chunck in response:
                        response_content = chunck["message"]["content"]
                        yield chunck["message"]["content"]
                
                stream = catch_response(response)
                st.write_stream(stream)

        with st.expander("Architecture recommendation"):
                response = ollama.chat(
                    model=md,
                    messages=[{
                        "role": "user", 
                        "content": f"""
                        You are an expert on analyzing ml models, 
                        Generate an in detail architecture recommendation, or suggest some powerful existing architectures based on the info I give you:
                        The model is a {model_type} model with given params and results: 
                        {df_sorted.to_string()},
                        Best architecture found:
                        {json.dumps(best_arch, indent=2) if best_arch else 'N/A'}
                        Model data:
                        {model_data_info}
                        """
                    }], stream=True
                )
                res_content = ""
                def catch_response(response):
                    global response_content
                    for chunck in response:
                        response_content = chunck["message"]["content"]
                        yield chunck["message"]["content"]
                
                stream = catch_response(response)
                st.write_stream(stream)

        with st.expander("Training optimization"):
                response = ollama.chat(
                    model=md,
                    messages=[{
                        "role": "user", 
                        "content": f"""
                        You are an expert on analyzing ml models, 
                        Generate an in detail training optimization recommendations based on the info I give you:
                        The model is a {exp["model_type"]} model with given params and results: 
                        {df_sorted.to_string()},
                        Model data:
                        {exp["model_data"]}
                        """
                    }], stream=True
                )
                response_content = ""
                def catch_response(response):
                    global response_content
                    for chunck in response:
                        response_content = chunck["message"]["content"]
                        yield chunck["message"]["content"]
                
                stream = catch_response(response)
                st.write_stream(stream)
    
    elif page == "Export Results":
        st.header("Export Results")
        
        if df.empty:
            st.warning("No data available to export.")
            return
        
        st.subheader("Download Experiment Data")
        
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="dqnn_experiments.csv",
            mime="text/csv"
        )
        
        # JSON export
        if full_data:
            json_data = json.dumps(full_data, indent=2)
            st.download_button(
                label="Download Full Data as JSON",
                data=json_data,
                file_name="dqnn_full_data.json",
                mime="application/json"
            )
        
        # Best architecture export
        if best_arch:
            best_arch_json = json.dumps(best_arch, indent=2)
            st.download_button(
                label="Download Best Architecture",
                data=best_arch_json,
                file_name="best_architecture.json",
                mime="application/json"
            )
        
        st.subheader("Data Summary")
        st.write(f"Total experiments: {len(df)}")
        st.write(f"Data file size: {len(csv_data)} characters")
        
        if 'accuracy' in df.columns or 'r2_score' in df.columns:
            perf_col = 'accuracy' if 'accuracy' in df.columns else 'r2_score'
            st.write(f"Best performance: {df[perf_col].max():.4f}")
            st.write(f"Average performance: {df[perf_col].mean():.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("**DQNN-TorchTrack Dashboard** | Neural Architecture Search with Deep Reinforcement Learning")

if __name__ == '__main__':
    main()