"""
Layout module for the Fraud Detection EDA Dashboard.

This module defines the Dash application layout including all
HTML and Dash Component Composition (DCC) elements.
"""

from dash import dcc, html

from src.eda.dashboard.utils import FRAUD_COLOR, LEGIT_COLOR


def create_header() -> html.Div:
    """
    Create the header section of the dashboard.

    Returns
    -------
    html.Div
        Header component with title and description.

    Examples
    --------
    >>> header = create_header()
    >>> isinstance(header, html.Div)
    True
    """
    return html.Div(
        className='header',
        style={
            'backgroundColor': '#2C3E50',
            'padding': '20px',
            'marginBottom': '30px',
            'borderRadius': '10px',
        },
        children=[
            html.H1(
                '💳 Credit Card Fraud Detection - EDA Dashboard',
                style={
                    'color': 'white',
                    'marginTop': '0',
                    'marginBottom': '10px',
                    'textAlign': 'center',
                    'fontSize': '32px',
                    'fontWeight': 'bold',
                }
            ),
            html.P(
                'Interactive exploratory data analysis for fraud detection patterns',
                style={
                    'color': '#ECF0F1',
                    'textAlign': 'center',
                    'fontSize': '16px',
                    'margin': '0',
                }
            ),
        ]
    )


def create_summary_card(stats: dict) -> html.Div:
    """
    Create a summary statistics card component.

    Parameters
    ----------
    stats : dict
        Dictionary containing summary statistics with keys:
        - 'total_transactions', 'fraud_count', 'legit_count'
        - 'fraud_percentage', 'total_amount', 'avg_amount'

    Returns
    -------
    html.Div
        Summary card component displaying key statistics.

    Examples
    --------
    >>> stats = {'total_transactions': 1000, 'fraud_count': 50, ...}
    >>> card = create_summary_card(stats)
    """
    return html.Div(
        className='summary-card',
        style={
            'backgroundColor': '#ECF0F1',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '30px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        },
        children=[
            html.H3(
                '📊 Summary Statistics',
                style={
                    'marginTop': '0',
                    'marginBottom': '20px',
                    'color': '#2C3E50',
                    'textAlign': 'center',
                }
            ),
            html.Div(
                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'},
                children=[
                    # Total Transactions
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '150px',
                            'textAlign': 'center',
                            'padding': '15px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                        },
                        children=[
                            html.H4(
                                f"{stats['total_transactions']:,}",
                                style={
                                    'margin': '0',
                                    'fontSize': '28px',
                                    'color': '#2C3E50',
                                    'fontWeight': 'bold',
                                }
                            ),
                            html.P(
                                'Total Transactions',
                                style={'margin': '5px 0 0 0', 'color': '#7F8C8D', 'fontSize': '14px'}
                            ),
                        ]
                    ),
                    # Fraud Count
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '150px',
                            'textAlign': 'center',
                            'padding': '15px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                            'border': f'3px solid {FRAUD_COLOR}',
                        },
                        children=[
                            html.H4(
                                f"{stats['fraud_count']:,}",
                                style={
                                    'margin': '0',
                                    'fontSize': '28px',
                                    'color': FRAUD_COLOR,
                                    'fontWeight': 'bold',
                                }
                            ),
                            html.P(
                                f"Fraud ({stats['fraud_percentage']}%)",
                                style={'margin': '5px 0 0 0', 'color': '#7F8C8D', 'fontSize': '14px'}
                            ),
                        ]
                    ),
                    # Legitimate Count
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '150px',
                            'textAlign': 'center',
                            'padding': '15px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                            'border': f'3px solid {LEGIT_COLOR}',
                        },
                        children=[
                            html.H4(
                                f"{stats['legit_count']:,}",
                                style={
                                    'margin': '0',
                                    'fontSize': '28px',
                                    'color': LEGIT_COLOR,
                                    'fontWeight': 'bold',
                                }
                            ),
                            html.P(
                                'Legitimate',
                                style={'margin': '5px 0 0 0', 'color': '#7F8C8D', 'fontSize': '14px'}
                            ),
                        ]
                    ),
                    # Average Amount
                    html.Div(
                        style={
                            'flex': '1',
                            'minWidth': '150px',
                            'textAlign': 'center',
                            'padding': '15px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                        },
                        children=[
                            html.H4(
                                f"${stats['avg_amount']:.2f}",
                                style={
                                    'margin': '0',
                                    'fontSize': '28px',
                                    'color': '#2C3E50',
                                    'fontWeight': 'bold',
                                }
                            ),
                            html.P(
                                'Avg Amount',
                                style={'margin': '5px 0 0 0', 'color': '#7F8C8D', 'fontSize': '14px'}
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_filters() -> html.Div:
    """
    Create the interactive filters section.

    Returns
    -------
    html.Div
        Filters component with amount range slider and log scale toggle.

    Examples
    --------
    >>> filters = create_filters()
    """
    return html.Div(
        className='filters',
        style={
            'backgroundColor': '#ECF0F1',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '30px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        },
        children=[
            html.H3(
                '🔧 Interactive Filters',
                style={
                    'marginTop': '0',
                    'marginBottom': '20px',
                    'color': '#2C3E50',
                }
            ),
            html.Div(
                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px', 'alignItems': 'center'},
                children=[
                    # Amount Range Filter
                    html.Div(
                        style={'flex': '2', 'minWidth': '250px'},
                        children=[
                            html.Label(
                                'Amount Range ($)',
                                style={
                                    'fontWeight': 'bold',
                                    'color': '#2C3E50',
                                    'marginBottom': '10px',
                                    'display': 'block',
                                }
                            ),
                            dcc.RangeSlider(
                                id='amount-range-slider',
                                min=0,
                                max=1000,
                                step=10,
                                value=[0, 1000],
                                marks={
                                    0: '0',
                                    250: '250',
                                    500: '500',
                                    750: '750',
                                    1000: '1000',
                                },
                                tooltip={'placement': 'bottom', 'always_visible': False},
                            ),
                            html.Div(
                                id='amount-range-display',
                                style={
                                    'marginTop': '10px',
                                    'textAlign': 'center',
                                    'color': '#7F8C8D',
                                    'fontSize': '14px',
                                }
                            ),
                        ]
                    ),
                    # Log Scale Toggle
                    html.Div(
                        style={'flex': '1', 'minWidth': '150px'},
                        children=[
                            html.Label(
                                'Log Scale',
                                style={
                                    'fontWeight': 'bold',
                                    'color': '#2C3E50',
                                    'marginBottom': '10px',
                                    'display': 'block',
                                }
                            ),
                            dcc.Checklist(
                                id='log-scale-toggle',
                                options=[
                                    {'label': ' Enable', 'value': 'log'}
                                ],
                                value=[],
                                style={'textAlign': 'center'},
                            ),
                        ]
                    ),
                    # Export Button
                    html.Div(
                        style={'flex': '1', 'minWidth': '150px'},
                        children=[
                            html.Label(
                                'Export',
                                style={
                                    'fontWeight': 'bold',
                                    'color': '#2C3E50',
                                    'marginBottom': '10px',
                                    'display': 'block',
                                }
                            ),
                            html.Button(
                                '📥 Export Dashboard',
                                id='export-button',
                                n_clicks=0,
                                style={
                                    'width': '100%',
                                    'padding': '10px',
                                    'backgroundColor': '#3498DB',
                                    'color': 'white',
                                    'border': 'none',
                                    'borderRadius': '5px',
                                    'cursor': 'pointer',
                                    'fontSize': '14px',
                                    'fontWeight': 'bold',
                                }
                            ),
                            html.Div(
                                id='export-status',
                                style={
                                    'marginTop': '10px',
                                    'textAlign': 'center',
                                    'fontSize': '12px',
                                }
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_charts_grid() -> html.Div:
    """
    Create the grid layout for all charts.

    Returns
    -------
    html.Div
        Grid component containing placeholders for all charts.

    Examples
    --------
    >>> grid = create_charts_grid()
    """
    return html.Div(
        className='charts-grid',
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(2, 1fr)',
            'gap': '20px',
            'marginBottom': '30px',
        },
        children=[
            # Class Distribution
            html.Div(
                style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'},
                children=[
                    dcc.Graph(id='class-dist-chart')
                ]
            ),
            # Amount Histogram
            html.Div(
                style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px'},
                children=[
                    dcc.Graph(id='amount-hist-chart')
                ]
            ),
            # Correlation Heatmap
            html.Div(
                style={
                    'backgroundColor': 'white',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'gridColumn': '1 / -1',  # Span full width
                },
                children=[
                    dcc.Graph(id='correlation-heatmap')
                ]
            ),
            # Time Patterns
            html.Div(
                style={
                    'backgroundColor': 'white',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'gridColumn': '1 / -1',  # Span full width
                },
                children=[
                    dcc.Graph(id='time-patterns-chart')
                ]
            ),
            # PCA Scatter
            html.Div(
                style={
                    'backgroundColor': 'white',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'gridColumn': '1 / -1',  # Span full width
                },
                children=[
                    dcc.Graph(id='pca-scatter-chart')
                ]
            ),
        ]
    )


def create_dashboard_layout(stats: dict) -> html.Div:
    """
    Create the complete dashboard layout.

    Parameters
    ----------
    stats : dict
        Dictionary containing summary statistics.

    Returns
    -------
    html.Div
        Complete dashboard layout component.

    Examples
    --------
    >>> stats = {'total_transactions': 1000, 'fraud_count': 50, ...}
    >>> layout = create_dashboard_layout(stats)
    """
    return html.Div(
        style={
            'fontFamily': 'Arial, sans-serif',
            'backgroundColor': '#F5F6FA',
            'minHeight': '100vh',
            'padding': '20px',
            'maxWidth': '1400px',
            'margin': '0 auto',
        },
        children=[
            # Header
            create_header(),

            # Summary Statistics Card
            create_summary_card(stats),

            # Interactive Filters
            create_filters(),

            # Charts Grid
            create_charts_grid(),

            # Footer
            html.Div(
                style={
                    'textAlign': 'center',
                    'padding': '20px',
                    'color': '#7F8C8D',
                    'fontSize': '14px',
                },
                children=[
                    html.P('Fraud Detection EDA Dashboard | Built with Plotly Dash'),
                    html.P('© 2024 | Portfolio Project'),
                ]
            ),
        ]
    )
