"""
Background Tasks Page for vitalDSP Webapp

This page provides a comprehensive interface for monitoring and managing
background processing tasks, including the ability to view, cancel, and
monitor long-running operations.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from vitalDSP_webapp.layout.common import (
    create_progress_card,
    create_progress_list,
    create_progress_interval,
    create_progress_store,
)


def tasks_layout():
    """
    Create the background tasks monitoring page layout.
    
    Returns
    -------
    html.Div
        The complete tasks page layout
    """
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H2(
                        [
                            html.I(className="fas fa-tasks mr-2"),
                            "Background Tasks",
                        ],
                        className="mb-3",
                    ),
                    html.P(
                        "Monitor and manage background processing tasks, including filtering, "
                        "quality assessment, and feature extraction operations.",
                        className="text-muted",
                    ),
                ],
                className="mb-4",
            ),
            
            # Main content area
            dbc.Row(
                [
                    # Left Panel - Active Tasks
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Active Tasks", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Active tasks list
                                            html.Div(
                                                id="active-tasks-list",
                                                children=[
                                                    html.P(
                                                        "No active tasks",
                                                        className="text-muted text-center"
                                                    )
                                                ]
                                            ),
                                            # Refresh button
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-sync-alt mr-2"),
                                                            "Refresh"
                                                        ],
                                                        id="refresh-tasks-btn",
                                                        color="primary",
                                                        size="sm",
                                                        className="mt-3"
                                                    )
                                                ],
                                                className="text-center"
                                            )
                                        ]
                                    )
                                ],
                                className="mb-4"
                            ),
                            
                            # Task Statistics
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Task Statistics", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H4("0", id="total-tasks-count", className="text-primary"),
                                                            html.P("Total Tasks", className="mb-0 text-muted")
                                                        ],
                                                        className="text-center"
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H4("0", id="running-tasks-count", className="text-warning"),
                                                            html.P("Running", className="mb-0 text-muted")
                                                        ],
                                                        className="text-center"
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H4("0", id="completed-tasks-count", className="text-success"),
                                                            html.P("Completed", className="mb-0 text-muted")
                                                        ],
                                                        className="text-center"
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H4("0", id="failed-tasks-count", className="text-danger"),
                                                            html.P("Failed", className="mb-0 text-muted")
                                                        ],
                                                        className="text-center"
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        md=6
                    ),
                    
                    # Right Panel - Task Details
                    dbc.Col(
                        [
                            # Selected Task Details
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Task Details", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="task-details-content",
                                                children=[
                                                    html.P(
                                                        "Select a task to view details",
                                                        className="text-muted text-center"
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                className="mb-4"
                            ),
                            
                            # Task History
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Recent Tasks", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="task-history-list",
                                                children=[
                                                    html.P(
                                                        "No recent tasks",
                                                        className="text-muted text-center"
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        md=6
                    )
                ]
            ),
            
            # Control Panel
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Task Controls", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(className="fas fa-play mr-2"),
                                                                    "Start New Task"
                                                                ],
                                                                id="start-new-task-btn",
                                                                color="success",
                                                                size="lg",
                                                                className="mb-2"
                                                            )
                                                        ],
                                                        md=4
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(className="fas fa-stop mr-2"),
                                                                    "Cancel All"
                                                                ],
                                                                id="cancel-all-tasks-btn",
                                                                color="danger",
                                                                size="lg",
                                                                className="mb-2"
                                                            )
                                                        ],
                                                        md=4
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                [
                                                                    html.I(className="fas fa-trash mr-2"),
                                                                    "Clear History"
                                                                ],
                                                                id="clear-history-btn",
                                                                color="secondary",
                                                                size="lg",
                                                                className="mb-2"
                                                            )
                                                        ],
                                                        md=4
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ],
                className="mt-4"
            ),
            
            # Progress Components
            create_progress_card(
                card_id="task-progress-card",
                title="Task Progress",
                show_details=True,
                show_cancel_button=True
            ),
            
            # Stores and Intervals
            create_progress_interval(
                interval_id="tasks-refresh-interval",
                interval_ms=2000,
                max_intervals=-1,
                disabled=False
            ),
            create_progress_store(
                store_id="tasks-store",
                initial_data={}
            ),
            create_progress_store(
                store_id="selected-task-store",
                initial_data={}
            ),
            
            # Hidden stores for task management
            dcc.Store(id="task-queue-store", data={}),
            dcc.Store(id="task-history-store", data=[]),
        ],
        style={"padding": "20px"},
    )


def create_task_item(task_info: dict) -> dbc.Card:
    """
    Create a task item card for display in the tasks list.
    
    Args:
        task_info: Dictionary containing task information
        
    Returns:
        dbc.Card containing the task item
    """
    task_id = task_info.get('task_id', 'unknown')
    operation_name = task_info.get('operation_name', 'Unknown Operation')
    status = task_info.get('status', 'unknown')
    progress = task_info.get('progress_percentage', 0)
    current_step = task_info.get('current_step', 'Unknown')
    
    # Status color mapping
    status_colors = {
        'pending': 'secondary',
        'running': 'warning',
        'completed': 'success',
        'failed': 'danger',
        'cancelled': 'dark'
    }
    
    status_color = status_colors.get(status, 'secondary')
    
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    # Task header
                    html.Div(
                        [
                            html.H6(
                                operation_name,
                                className="mb-1",
                                style={"font-weight": "bold"}
                            ),
                            dbc.Badge(
                                status.title(),
                                color=status_color,
                                className="float-right"
                            )
                        ],
                        className="mb-2"
                    ),
                    
                    # Progress bar
                    dbc.Progress(
                        value=progress,
                        max=100,
                        animated=(status == 'running'),
                        striped=(status == 'running'),
                        color=status_color,
                        className="mb-2"
                    ),
                    
                    # Current step
                    html.P(
                        current_step,
                        className="mb-1 text-muted small"
                    ),
                    
                    # Task ID and controls
                    html.Div(
                        [
                            html.Small(
                                f"ID: {task_id[:8]}...",
                                className="text-muted"
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Details",
                                        id=f"task-details-btn-{task_id}",
                                        color="outline-primary",
                                        size="sm",
                                        className="mr-1"
                                    ),
                                    dbc.Button(
                                        "Cancel",
                                        id=f"task-cancel-btn-{task_id}",
                                        color="outline-danger",
                                        size="sm"
                                    ) if status == 'running' else None
                                ],
                                className="float-right"
                            )
                        ],
                        className="d-flex justify-content-between align-items-center"
                    )
                ]
            )
        ],
        className="mb-2",
        style={"cursor": "pointer"},
        id=f"task-item-{task_id}"
    )


def create_task_details(task_info: dict) -> html.Div:
    """
    Create detailed task information display.
    
    Args:
        task_info: Dictionary containing detailed task information
        
    Returns:
        html.Div containing the task details
    """
    if not task_info:
        return html.P("No task selected", className="text-muted")
    
    task_id = task_info.get('task_id', 'unknown')
    operation_name = task_info.get('operation_name', 'Unknown Operation')
    status = task_info.get('status', 'unknown')
    progress = task_info.get('progress_percentage', 0)
    current_step = task_info.get('current_step', 'Unknown')
    start_time = task_info.get('start_time', 'Unknown')
    end_time = task_info.get('end_time', None)
    error_message = task_info.get('error_message', None)
    metadata = task_info.get('metadata', {})
    
    details = [
        html.H6("Task Information", className="mb-3"),
        html.Div(
            [
                html.Strong("Operation: "),
                html.Span(operation_name)
            ],
            className="mb-2"
        ),
        html.Div(
            [
                html.Strong("Status: "),
                dbc.Badge(status.title(), color="primary")
            ],
            className="mb-2"
        ),
        html.Div(
            [
                html.Strong("Progress: "),
                html.Span(f"{progress:.1f}%")
            ],
            className="mb-2"
        ),
        html.Div(
            [
                html.Strong("Current Step: "),
                html.Span(current_step)
            ],
            className="mb-2"
        ),
        html.Div(
            [
                html.Strong("Started: "),
                html.Span(str(start_time))
            ],
            className="mb-2"
        ),
    ]
    
    if end_time:
        details.append(
            html.Div(
                [
                    html.Strong("Completed: "),
                    html.Span(str(end_time))
                ],
                className="mb-2"
            )
        )
    
    if error_message:
        details.append(
            html.Div(
                [
                    html.Strong("Error: "),
                    html.Span(error_message, className="text-danger")
                ],
                className="mb-2"
            )
        )
    
    if metadata:
        details.extend([
            html.Hr(),
            html.H6("Metadata", className="mb-3"),
            html.Pre(
                str(metadata),
                className="bg-light p-2 rounded",
                style={"font-size": "0.8rem"}
            )
        ])
    
    return html.Div(details)
