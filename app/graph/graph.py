from langgraph.graph import END, START, StateGraph

from app.graph.nodes import PlannerNode, ResponseGeneratorNode, ToolExecutorNode
from app.graph.state import RuntimeState
from app.model_gateway.client import ModelGatewayClient
from app.planner.planner import PlannerService
from app.prompts.builder import PromptBuilder
from app.tool_executor.service import ToolExecutorService


class WorkflowManager:
    def __init__(
        self,
        planner_service: PlannerService,
        model_gateway_client: ModelGatewayClient,
        tool_executor_service: ToolExecutorService,
    ) -> None:
        self._planner_node = PlannerNode(planner_service)
        self._tool_executor_node = ToolExecutorNode(tool_executor_service)
        self._response_generator_node = ResponseGeneratorNode(PromptBuilder(), model_gateway_client)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RuntimeState)
        graph.add_node('planner', self._planner_node)
        graph.add_node('tool_executor', self._tool_executor_node)
        graph.add_node('response_generator', self._response_generator_node)
        graph.add_edge(START, 'planner')
        graph.add_edge('planner', 'tool_executor')
        graph.add_edge('tool_executor', 'response_generator')
        graph.add_edge('response_generator', END)
        return graph.compile()

    async def run(self, state: RuntimeState) -> RuntimeState:
        final_state = await self._graph.ainvoke(state)
        if isinstance(final_state, dict):
            return RuntimeState(**final_state)
        return final_state
