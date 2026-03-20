"""Planner agent that talks to VLM providers and returns executable plans."""

import os
import json
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.chat_prompts import (
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_PROMPT_PLANNER_OBJECT_DESCRIBER,
    SYSTEM_PROMPT_PLANNER_REPLAN
)
from dronebuddylib.atoms.planning.planner_models import ActionPlan, PlannerAction, PlannerActionType
from dronebuddylib.atoms.planning.vlm_client import (
    BaseVLMClient,
    create_vlm_client,
    VLMProvider,
    VLMResponse
)

if TYPE_CHECKING:
    from dronebuddylib.atoms.planning.planner_configs import PlannerConfigs
    from dronebuddylib.atoms.planning.session_logger import SessionLogger

logger = Logger()


class PlannerAgent:
    """
    Generates and regenerates search plans using a configured VLM backend.
    """
    
    def __init__(
        self, 
        provider: str = "openai",
        api_key: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            provider: VLM provider ("openai", "anthropic", "google")
            api_key: API key for the provider
            model: Model to use (uses provider default if not specified)
            temperature: Model temperature (lower = more deterministic)
        """
        self.provider = provider
        self.temperature = temperature
        
        # Create VLM client
        self.vlm_client: BaseVLMClient = create_vlm_client(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Prompts are defined centrally in chat_prompts.py.
        self.system_prompt = SYSTEM_PROMPT_PLANNER
        self.object_describer_prompt = SYSTEM_PROMPT_PLANNER_OBJECT_DESCRIBER
        self.replanning_prompt = SYSTEM_PROMPT_PLANNER_REPLAN
        
        self.vlm_client.set_system_prompt(self.system_prompt)
        
        # Injected by PlannerExecutor at the beginning of a session.
        self.session_logger: Optional['SessionLogger'] = None
        
        logger.log_info('PlannerAgent', 
            f'Initialized with provider: {provider}, model: {self.vlm_client.model}')
    
    @classmethod
    def from_config(cls, config: 'PlannerConfigs') -> 'PlannerAgent':
        """
        Create a PlannerAgent from a PlannerConfigs object.
        
        Args:
            config: PlannerConfigs instance with VLM settings
            
        Returns:
            Configured PlannerAgent instance
        """
        return cls(
            provider=config.vlm_provider,
            api_key=config.vlm_api_key,
            model=config.vlm_model if config.vlm_model else None,
            temperature=config.vlm_temperature
        )
    
    def generate_plan(
        self, 
        user_request: str, 
        waypoint_names: List[str],
        current_waypoint: str = "START"
    ) -> Optional[ActionPlan]:
        """
        Generate an action plan based on user request and available waypoints.
        
        Args:
            user_request: Natural language request from user (e.g., "Find a red cup")
            waypoint_names: List of available waypoint names
            current_waypoint: Drone's current waypoint position
            
        Returns:
            ActionPlan object containing ordered actions, or None if generation failed
        """
        logger.log_info('PlannerAgent', f'Generating plan for request: "{user_request}"')
        logger.log_debug('PlannerAgent', f'Available waypoints: {waypoint_names}')
        
        # Ensure we're using the planning system prompt
        self.vlm_client.set_system_prompt(self.system_prompt)
        self.vlm_client.clear_history()
        
        # Build the user message with context
        user_message = self._build_planning_message(user_request, waypoint_names, current_waypoint)
        
        try:
            # Capture request context for session logs.
            _history_snapshot = [
                {"role": msg.role, "content": msg.content}
                for msg in self.vlm_client.conversation_history
                if msg.role != "system"
            ]
            _active_system_prompt = self.vlm_client.system_prompt or self.system_prompt
            
            _t0 = time.time()
            response: VLMResponse = self.vlm_client.send_message(user_message)
            _latency = time.time() - _t0
            
            if self.session_logger and response:
                from dronebuddylib.atoms.planning.session_logger import SessionLogger
                self.session_logger.record_vlm_call(
                    call_type="Plan Generation",
                    system_prompt=_active_system_prompt,
                    conversation_history=_history_snapshot,
                    user_message=user_message,
                    response_content=response.content,
                    latency_seconds=_latency,
                )
            
            if response and response.content:
                plan = self._parse_plan_response(response.content)
                if plan:
                    logger.log_success('PlannerAgent', f'Generated plan with {len(plan.actions)} actions')
                    return plan
                else:
                    logger.log_error('PlannerAgent', 'Failed to parse plan response')
                    return None
            else:
                logger.log_error('PlannerAgent', 'Empty response from VLM')
                return None
                
        except Exception as e:
            logger.log_error('PlannerAgent', f'Plan generation failed: {e}')
            return None
    
    def regenerate_plan(
        self,
        target_object: str,
        waypoint_names: List[str],
        current_waypoint: str,
        visited_waypoints: List[str],
        user_feedback: str = "",
        is_coco_class: bool = True,
        related_objects: List[str] = None
    ) -> Optional[ActionPlan]:
        """
        Regenerate an action plan after a failed search or user rejection.
        
        Args:
            target_object: The object being searched for
            waypoint_names: List of all available waypoint names
            current_waypoint: Drone's current waypoint position
            visited_waypoints: List of waypoints already visited
            user_feedback: Optional feedback from user about why they rejected
            is_coco_class: Whether the target is a COCO class (uses YOLO vs YOLO-World)
            related_objects: For non-COCO objects, list of related detection terms
            
        Returns:
            New ActionPlan object, or None if generation failed
        """
        logger.log_info('PlannerAgent', f'Regenerating plan for: "{target_object}"')
        logger.log_debug('PlannerAgent', f'Already visited: {visited_waypoints}')
        
        if related_objects is None:
            related_objects = []
        
        # Clear history and set replanning prompt
        self.vlm_client.clear_history()
        self.vlm_client.set_system_prompt(self.replanning_prompt)
        
        # Build replanning message
        unvisited = [wp for wp in waypoint_names if wp not in visited_waypoints]
        
        # Include detection mode so replans stay consistent.
        detection_mode = "Standard YOLO (COCO class)" if is_coco_class else "YOLO-World (open-vocabulary)"
        related_info = f"\n- **Related Objects for Detection**: {', '.join(related_objects)}" if related_objects else ""
        
        user_message = f"""
## Current Situation
- **Target Object**: {target_object}
- **Is COCO Class**: {is_coco_class}
- **Detection Mode**: {detection_mode}{related_info}
- **Drone's Current Position**: {current_waypoint}
- **Waypoints Already Visited**: {', '.join(visited_waypoints) if visited_waypoints else 'None'}
- **Waypoints Not Yet Visited**: {', '.join(unvisited) if unvisited else 'All waypoints have been visited'}
- **All Available Waypoints**: {', '.join(waypoint_names)}

## User Feedback
{user_feedback if user_feedback else 'The user wants to continue searching.'}

Please generate a new search plan. Keep the same is_coco_class value and related_objects unless user feedback suggests otherwise.
"""
        
        try:
            _history_snapshot = [
                {"role": msg.role, "content": msg.content}
                for msg in self.vlm_client.conversation_history
                if msg.role != "system"
            ]
            _active_system_prompt = self.vlm_client.system_prompt or self.replanning_prompt
            
            _t0 = time.time()
            response: VLMResponse = self.vlm_client.send_message(user_message)
            _latency = time.time() - _t0
            
            if self.session_logger and response:
                from dronebuddylib.atoms.planning.session_logger import SessionLogger
                self.session_logger.record_vlm_call(
                    call_type="Plan Regeneration",
                    system_prompt=_active_system_prompt,
                    conversation_history=_history_snapshot,
                    user_message=user_message,
                    response_content=response.content,
                    latency_seconds=_latency,
                )
            
            if response and response.content:
                plan = self._parse_plan_response(response.content)
                if plan:
                    logger.log_success('PlannerAgent', f'Regenerated plan with {len(plan.actions)} actions')
                    return plan
            
            logger.log_error('PlannerAgent', 'Failed to regenerate plan')
            return None
            
        except Exception as e:
            logger.log_error('PlannerAgent', f'Plan regeneration failed: {e}')
            return None
    
    def describe_object_in_images(
        self,
        target_object: str,
        image_paths: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Use VLM to describe the detected object in the provided images for user confirmation.
        
        This method is called when YOLO detects the target object. The VLM analyzes
        the actual image to provide a human-readable description that helps the user
        confirm if this is the item they were looking for.
        
        Args:
            target_object: The YOLO class name that was detected (e.g., "cup")
            image_paths: List of paths to images containing the detected object
            
        Returns:
            Dictionary with object description for user confirmation:
            {
                "object_description": "Detailed description of what the VLM sees",
                "visual_characteristics": ["color", "shape", "condition", ...],
                "location_context": "Where the object appears to be"
            }
        """
        logger.log_info('PlannerAgent', f'Describing detected {target_object} from {len(image_paths)} images')
        
        # Clear history and set object describer prompt
        self.vlm_client.clear_history()
        self.vlm_client.set_system_prompt(self.object_describer_prompt)
        
        user_message = f"""I found a "{target_object}" in the image. Please describe what you see in detail so the user can confirm if this is the item they were looking for.

The user was searching for this object. Describe:
1. What the object looks like (color, size, shape)
2. Its condition (new, used, clean, dirty, etc.)
3. Where it appears to be located in the scene
4. Any distinguishing features that would help identify it
"""
        
        try:
            # Use the first available image path (top frames are already confidence-sorted).
            valid_image = None
            for image_path in image_paths[:3]:  # Try up to 3 images
                if os.path.exists(image_path):
                    valid_image = image_path
                    break
            
            if not valid_image:
                logger.log_warning('PlannerAgent', 'No valid images found for description')
                return None
            
            logger.log_info('PlannerAgent', f'Sending image to VLM for description: {valid_image}')
            _history_snapshot = [
                {"role": msg.role, "content": msg.content}
                for msg in self.vlm_client.conversation_history
                if msg.role != "system"
            ]
            _active_system_prompt = self.vlm_client.system_prompt or self.object_describer_prompt
            
            _t0 = time.time()
            response: VLMResponse = self.vlm_client.send_message(user_message, valid_image)
            _latency = time.time() - _t0
            
            if self.session_logger and response:
                from dronebuddylib.atoms.planning.session_logger import SessionLogger
                self.session_logger.record_vlm_call(
                    call_type="Object Description",
                    system_prompt=_active_system_prompt,
                    conversation_history=_history_snapshot,
                    user_message=user_message,
                    response_content=response.content,
                    latency_seconds=_latency,
                    image_path=valid_image,
                )
            
            if response and response.content:
                try:
                    content = response.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    return json.loads(content.strip())
                except json.JSONDecodeError:
                    return {
                        "object_description": response.content,
                        "visual_characteristics": [],
                        "location_context": ""
                    }
            
            return None
            
        except Exception as e:
            logger.log_error('PlannerAgent', f'Object description failed: {e}')
            return None
    
    def _build_planning_message(
        self, 
        user_request: str, 
        waypoint_names: List[str],
        current_waypoint: str
    ) -> str:
        """Build the user message for plan generation."""
        return f"""
## User Request
"{user_request}"

## Available Waypoints
The following waypoints are available for navigation:
{', '.join(waypoint_names)}

## Current Position
The drone is currently at: {current_waypoint}

Please generate an optimal action plan to find the requested object.
"""
    
    def _parse_plan_response(self, response_content: str) -> Optional[ActionPlan]:
        """Parse the VLM response into an ActionPlan object."""
        try:
            content = response_content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            actions = []
            for action_data in data.get('actions', []):
                action_type_str = action_data.get('action_type', '')
                
                if action_type_str == 'navigate':
                    action_type_str = 'navigate_to_waypoint'
                elif action_type_str == 'scan':
                    action_type_str = 'scan_area'
                
                try:
                    action_type = PlannerActionType(action_type_str)
                except ValueError:
                    logger.log_warning('PlannerAgent', f'Unknown action type: {action_type_str}')
                    continue
                
                waypoint_name = (
                    action_data.get('waypoint_name') or 
                    action_data.get('destination') or 
                    action_data.get('waypoint')
                )
                
                action = PlannerAction(
                    action_type=action_type,
                    waypoint_name=waypoint_name,
                    reason=action_data.get('reason', action_data.get('description', ''))
                )
                actions.append(action)
            
            plan = ActionPlan(
                target_object=data.get('target_object', ''),
                actions=actions,
                is_coco_class=data.get('is_coco_class', True),  # Default True for backward compatibility
                related_objects=data.get('related_objects', []),  # Empty list if not provided
                user_description=data.get('user_description', ''),
                reasoning=data.get('reasoning', '')
            )
            
            if plan.is_coco_class:
                logger.log_debug('PlannerAgent', f'Using COCO detection for: {plan.target_object}')
            else:
                detection_targets = plan.get_detection_targets()
                logger.log_debug('PlannerAgent', f'Using YOLO-World detection for: {detection_targets}')
            
            return plan
            
        except json.JSONDecodeError as e:
            logger.log_error('PlannerAgent', f'JSON parse error: {e}')
            logger.log_debug('PlannerAgent', f'Response content: {response_content[:500]}')
            return None
        except Exception as e:
            logger.log_error('PlannerAgent', f'Plan parsing error: {e}')
            return None
    
    def get_provider_name(self) -> str:
        """Get the current VLM provider name."""
        return self.provider
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.vlm_client.model
    
    def reset_session(self):
        """Reset the VLM session and restore the planning system prompt."""
        self.vlm_client.clear_history()
        self.vlm_client.set_system_prompt(self.system_prompt)
        logger.log_debug('PlannerAgent', 'Session reset')
