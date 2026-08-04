from app.graph.state import RuntimeState

__all__ = ['WorkflowManager', 'RuntimeState']


def __getattr__(name: str):
    if name == 'WorkflowManager':
        from app.graph.graph import WorkflowManager

        return WorkflowManager
    raise AttributeError(name)
