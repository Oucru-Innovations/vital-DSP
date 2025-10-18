"""
Background Tasks Callbacks for vitalDSP Webapp

This module handles callbacks for the background tasks monitoring page,
including task management, progress tracking, and cancellation.
"""

from dash.dependencies import Input, Output, State
from dash import html, no_update, callback_context
import logging
from datetime import datetime
from typing import Dict, Any, List

# Import services
from vitalDSP_webapp.services.progress_tracker import get_progress_tracker
from vitalDSP_webapp.layout.pages.tasks_page import create_task_item, create_task_details

logger = logging.getLogger(__name__)


def register_tasks_callbacks(app):
    """
    Register all background tasks-related callbacks.
    
    Parameters
    ----------
    app : Dash
        The Dash application instance
    """
    
    @app.callback(
        [
            Output("active-tasks-list", "children"),
            Output("total-tasks-count", "children"),
            Output("running-tasks-count", "children"),
            Output("completed-tasks-count", "children"),
            Output("failed-tasks-count", "children"),
        ],
        [
            Input("tasks-refresh-interval", "n_intervals"),
            Input("refresh-tasks-btn", "n_clicks"),
        ]
    )
    def update_tasks_display(n_intervals, refresh_clicks):
        """
        Update the tasks display with current task information.
        """
        try:
            progress_tracker = get_progress_tracker()
            all_tasks = progress_tracker.get_all_active_tasks()
            
            if not all_tasks:
                return (
                    [html.P("No active tasks", className="text-muted text-center")],
                    "0", "0", "0", "0"
                )
            
            # Create task items
            task_items = []
            for task in all_tasks:
                task_info = {
                    'task_id': task.task_id,
                    'operation_name': task.operation_name,
                    'status': task.status.value,
                    'progress_percentage': task.progress_percentage,
                    'current_step': task.current_step,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'error_message': task.error_message,
                    'metadata': task.metadata
                }
                task_items.append(create_task_item(task_info))
            
            # Count tasks by status
            total_count = len(all_tasks)
            running_count = sum(1 for task in all_tasks if task.status.value == 'running')
            completed_count = sum(1 for task in all_tasks if task.status.value == 'completed')
            failed_count = sum(1 for task in all_tasks if task.status.value == 'failed')
            
            return (
                task_items,
                str(total_count),
                str(running_count),
                str(completed_count),
                str(failed_count)
            )
            
        except Exception as e:
            logger.error(f"Error updating tasks display: {e}")
            return (
                [html.P("Error loading tasks", className="text-danger text-center")],
                "0", "0", "0", "0"
            )
    
    @app.callback(
        Output("task-details-content", "children"),
        [
            Input("active-tasks-list", "children"),
        ],
        [
            State("selected-task-store", "data")
        ]
    )
    def update_task_details(task_items, selected_task_data):
        """
        Update the task details panel.
        """
        if not selected_task_data or not selected_task_data.get('task_id'):
            return html.P("Select a task to view details", className="text-muted text-center")
        
        return create_task_details(selected_task_data)
    
    @app.callback(
        Output("selected-task-store", "data"),
        [
            Input("active-tasks-list", "children"),
        ],
        prevent_initial_call=True
    )
    def handle_task_selection(task_items):
        """
        Handle task selection from the tasks list.
        """
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update
        
        # This callback will be triggered by individual task buttons
        # We'll handle the actual selection in a separate callback
        return no_update
    
    @app.callback(
        [
            Output("task-progress-card", "style"),
            Output("task-progress-card-progress", "value"),
            Output("task-progress-card-progress-status", "children"),
            Output("task-progress-card-details", "children"),
        ],
        [
            Input("tasks-refresh-interval", "n_intervals"),
        ],
        [
            State("selected-task-store", "data")
        ]
    )
    def update_task_progress(n_intervals, selected_task_data):
        """
        Update the task progress card.
        """
        if not selected_task_data or not selected_task_data.get('task_id'):
            return (
                {"display": "none"},
                0,
                "No task selected",
                []
            )
        
        try:
            progress_tracker = get_progress_tracker()
            task_id = selected_task_data['task_id']
            task_progress = progress_tracker.get_task_progress(task_id)
            
            if not task_progress:
                return (
                    {"display": "none"},
                    0,
                    "Task not found",
                    []
                )
            
            # Calculate time elapsed
            start_time = task_progress.start_time
            current_time = datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            
            details = [
                html.P(f"Operation: {task_progress.operation_name}", className="mb-1"),
                html.P(f"Status: {task_progress.status.value.title()}", className="mb-1"),
                html.P(f"Time elapsed: {elapsed_time:.1f}s", className="mb-0")
            ]
            
            return (
                {"display": "block"},
                task_progress.progress_percentage,
                task_progress.current_step,
                details
            )
            
        except Exception as e:
            logger.error(f"Error updating task progress: {e}")
            return (
                {"display": "none"},
                0,
                "Error loading progress",
                []
            )
    
    @app.callback(
        Output("task-history-list", "children"),
        [
            Input("tasks-refresh-interval", "n_intervals"),
        ]
    )
    def update_task_history(n_intervals):
        """
        Update the task history list.
        """
        try:
            progress_tracker = get_progress_tracker()
            all_tasks = progress_tracker.get_all_active_tasks()
            
            # Filter completed/failed tasks for history
            history_tasks = [
                task for task in all_tasks 
                if task.status.value in ['completed', 'failed', 'cancelled']
            ]
            
            if not history_tasks:
                return [html.P("No recent tasks", className="text-muted text-center")]
            
            # Sort by end time (most recent first)
            history_tasks.sort(key=lambda x: x.end_time or datetime.min, reverse=True)
            
            # Show only last 10 tasks
            history_tasks = history_tasks[:10]
            
            history_items = []
            for task in history_tasks:
                status_color = {
                    'completed': 'success',
                    'failed': 'danger',
                    'cancelled': 'dark'
                }.get(task.status.value, 'secondary')
                
                history_items.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(task.operation_name),
                                    dbc.Badge(
                                        task.status.value.title(),
                                        color=status_color,
                                        size="sm",
                                        className="ml-2"
                                    )
                                ],
                                className="d-flex justify-content-between align-items-center mb-1"
                            ),
                            html.Small(
                                f"Completed: {task.end_time.strftime('%H:%M:%S') if task.end_time else 'Unknown'}",
                                className="text-muted"
                            )
                        ],
                        className="border-bottom pb-2 mb-2"
                    )
                )
            
            return history_items
            
        except Exception as e:
            logger.error(f"Error updating task history: {e}")
            return [html.P("Error loading history", className="text-danger text-center")]
    
    @app.callback(
        Output("tasks-store", "data"),
        [
            Input("cancel-all-tasks-btn", "n_clicks"),
            Input("clear-history-btn", "n_clicks"),
        ]
    )
    def handle_task_controls(cancel_clicks, clear_clicks):
        """
        Handle task control buttons.
        """
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        try:
            progress_tracker = get_progress_tracker()
            
            if trigger_id == "cancel-all-tasks-btn":
                # Cancel all running tasks
                all_tasks = progress_tracker.get_all_active_tasks()
                cancelled_count = 0
                
                for task in all_tasks:
                    if task.status.value == 'running':
                        if progress_tracker.cancel_task(task.task_id):
                            cancelled_count += 1
                
                logger.info(f"Cancelled {cancelled_count} tasks")
                
            elif trigger_id == "clear-history-btn":
                # Clean up completed tasks
                progress_tracker.cleanup_completed_tasks(max_age_hours=0)
                logger.info("Cleared task history")
            
            return {"action": trigger_id, "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error handling task control {trigger_id}: {e}")
            return {"error": str(e)}
    
    # Dynamic callbacks for individual task actions
    def create_task_action_callbacks():
        """
        Create dynamic callbacks for individual task actions.
        This would be called when tasks are created to handle
        task-specific buttons like details and cancel.
        """
        pass
    
    logger.info("Tasks callbacks registered successfully")
