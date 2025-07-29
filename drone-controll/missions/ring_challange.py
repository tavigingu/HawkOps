import asyncio
import logging
from typing import List, TYPE_CHECKING
from datatypes import MissionResult, MissionStatus, Position
from mission_base import Mission

# Evită importarea circulară
if TYPE_CHECKING:
    from drone_controller import ParrotAnafi

logger = logging.getLogger(__name__)

class RingChallengeMission(Mission):
    """Side mission: Ring Challenge - pass through rings"""
    
    def __init__(self, ring_positions: List[Position], speed: float = 2.0):
        super().__init__("Ring Challenge", "Pass through rings in order")
        self.ring_positions = ring_positions
        self.speed = speed
        self.current_ring_index = 0
        self.completed_rings = 0
    
    async def execute(self, drone: 'ParrotAnafi') -> MissionResult:
        self.status = MissionStatus.EXECUTING
        logger.info(f"Starting Ring Challenge with {len(self.ring_positions)} rings")
        
        try:
            for i, ring_pos in enumerate(self.ring_positions):
                await self._wait_if_paused()
                
                if self.status == MissionStatus.FAILED:
                    break
                
                self.current_ring_index = i
                self.progress = i / len(self.ring_positions)
                
                logger.info(f"Navigating to ring {i+1}/{len(self.ring_positions)}")
                
                success = await self._navigate_through_ring(drone, ring_pos, i+1)
                if success:
                    self.completed_rings += 1
                else:
                    logger.warning(f"Failed at ring {i+1}")
            
            self.status = MissionStatus.COMPLETED
            self.progress = 1.0
            
            return MissionResult(
                success=True,
                message=f"Ring Challenge completed! Passed through {self.completed_rings}/{len(self.ring_positions)} rings",
                data={
                    "completed_rings": self.completed_rings,
                    "total_rings": len(self.ring_positions),
                    "success_rate": self.completed_rings / len(self.ring_positions)
                }
            )
            
        except Exception as e:
            self.status = MissionStatus.FAILED
            logger.error(f"Error during Ring Challenge: {e}")
            return MissionResult(success=False, message=f"Error during Ring Challenge: {str(e)}")
    
    async def _navigate_through_ring(self, drone: 'ParrotAnafi', ring_pos: Position, ring_number: int) -> bool:
        """Navigate through the specified ring"""
        try:
            # Approach position
            approach_pos = Position(ring_pos.x - 2, ring_pos.y, ring_pos.z)
            await drone.move_to_position(approach_pos)
            
            # Move through the ring
            await drone.move_to_position(ring_pos)
            
            # Exit position
            exit_pos = Position(ring_pos.x + 2, ring_pos.y, ring_pos.z)
            await drone.move_to_position(exit_pos)
            
            logger.info(f"Successfully passed ring {ring_number}!")
            return True
            
        except Exception as e:
            logger.error(f"Error while navigating through ring {ring_number}: {e}")
            return False