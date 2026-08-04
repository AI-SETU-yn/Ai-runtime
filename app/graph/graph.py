from langgraph.graph import END, START, StateGraph

from app.agent.agent import GeneralAgent
from app.agent.models import AgentState
from app.agent.tool_discovery import ToolDiscovery
from app.graph.nodes import CurrentInfoRouterNode, ResponseGeneratorNode
from app.graph.state import RuntimeState
from app.model_gateway.adapter_resolver import ModelGatewayAdapterResolver
from app.model_gateway.client import ModelGatewayClient
from app.planner.planner import PlannerService
from app.rbac.service import RBACAuthorizationService
from app.tool_executor.service import ToolExecutorService
from app.tool_registry.service import ToolRegistryService
from app.web_search.client import WebSearchClient


class WorkflowManager:
    def __init__(
        self,
        planner_service: PlannerService,
        model_gateway_client: ModelGatewayClient,
        tool_executor_service: ToolExecutorService,
        tool_registry_service: ToolRegistryService | None = None,
        web_search_client: WebSearchClient | None = None,
        rbac_authorization_service: RBACAuthorizationService | None = None,
    ) -> None:
        self._response_generator_node = ResponseGeneratorNode(
            model_gateway_client,
            ModelGatewayAdapterResolver(tool_registry_service),
        )
        self._general_agent = GeneralAgent(
            planner_service,
            tool_executor_service,
            self._response_generator_node,
            tool_discovery=ToolDiscovery(tool_registry_service),
            current_info_router=CurrentInfoRouterNode(web_search_client),
            rbac_authorization_service=rbac_authorization_service,
        )
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node('reason', self._general_agent.reason)
        graph.add_node('act', self._general_agent.act)
        graph.add_node('observe', self._general_agent.observe)
        graph.add_node('decide', self._general_agent.decide)
        graph.add_node('response_generator', self._general_agent.generate_final_response)
        graph.add_edge(START, 'reason')
        graph.add_edge('reason', 'act')
        graph.add_edge('act', 'observe')
        graph.add_edge('observe', 'decide')
        graph.add_conditional_edges(
            'decide',
            self._general_agent.route_after_decision,
            {
                'continue': 'reason',
                'finalize': 'response_generator',
            },
        )
        graph.add_edge('response_generator', END)
        return graph.compile()

    async def run(self, state: RuntimeState) -> RuntimeState:
        final_state = await self._graph.ainvoke(AgentState.from_runtime_state(state))
        if isinstance(final_state, dict):
            return AgentState(**final_state)
        return final_state
