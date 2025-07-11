import asyncio
import logging
from typing import List, Optional, Dict, Callable
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveTo
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from mission_manager import MissionManager, MissionResult
from datatypes import Position, DroneState
from missions.ring_challange import RingChallengeMission
from missions.room_search import RoomSearchMission
from mission_base import Mission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParrotAnafi:
    """Main class for controlling the Parrot Anafi drone"""
    
    def __init__(self, drone_ip: str = "192.168.42.1"):
        self.drone_ip = drone_ip
        self.drone = olympe.Drone(drone_ip)
        self.state = DroneState.DISCONNECTED
        self.mission_manager = MissionManager()
        self.current_position = Position(0, 0, 0)
        self.is_flying = False
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Register default missions
        self._register_default_missions()
    
    def _register_default_missions(self):
        """Registers default missions"""
        # Main mission
        room_search = RoomSearchMission(["Living Room", "Kitchen", "Bedroom"])
        self.mission_manager.register_mission(room_search)
        
        # Ring Challenge
        ring_positions = [
            Position(5, 0, 2),
            Position(10, 5, 2),
            Position(15, 0, 2)
        ]
        ring_challenge = RingChallengeMission(ring_positions)
        self.mission_manager.register_mission(ring_challenge)
    
    async def connect(self) -> bool:
        """Connects to the drone"""
        try:
            self.drone.connect()
            self.state = DroneState.CONNECTED
            logger.info("Connected to drone")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnects from the drone"""
        if self.is_flying:
            await self.land()
        self.drone.disconnect()
        self.state = DroneState.DISCONNECTED
        logger.info("Disconnected from drone")
    
    async def takeoff(self) -> bool:
        """Takeoff"""
        try:
            self.drone(TakeOff()).wait()
            self.is_flying = True
            self.state = DroneState.FLYING
            logger.info("Takeoff successful")
            return True
        except Exception as e:
            logger.error(f"Takeoff error: {e}")
            return False
    
    async def land(self) -> bool:
        """Landing"""
        try:
            self.drone(Landing()).wait()
            self.is_flying = False
            self.state = DroneState.CONNECTED
            logger.info("Landing successful")
            return True
        except Exception as e:
            logger.error(f"Landing error: {e}")
            return False
    
    async def move_to_position(self, position: Position):
        """Move to a specific position"""
        try:
            self.drone(moveTo(position.x, position.y, position.z, position.yaw)).wait()
            self.current_position = position
            logger.info(f"Moved to position: ({position.x}, {position.y}, {position.z})")
        except Exception as e:
            logger.error(f"Movement error: {e}")
    
    async def move_to_altitude(self, altitude: float):
        """Move to a specific altitude"""
        new_pos = Position(
            self.current_position.x,
            self.current_position.y,
            altitude
        )
        await self.move_to_position(new_pos)
    
    # Mission Management Methods
    async def execute_mission(self, mission_name: str) -> MissionResult:
        """Executes a mission"""
        if not self.is_flying:
            await self.takeoff()
        
        return await self.mission_manager.execute_mission(mission_name, self)
    
    def get_available_missions(self) -> List[str]:
        """Returns available missions"""
        return self.mission_manager.get_available_missions()
    
    def get_mission_info(self, mission_name: str) -> Optional[Dict]:
        """Returns information about a mission"""
        return self.mission_manager.get_mission_info(mission_name)
    
    def get_current_mission_status(self) -> Optional[Dict]:
        """Returns current mission status"""
        return self.mission_manager.get_current_mission_status()
    
    async def pause_mission(self) -> bool:
        """Pauses the mission"""
        return await self.mission_manager.pause_current_mission()
    
    async def resume_mission(self) -> bool:
        """Resumes the mission"""
        return await self.mission_manager.resume_current_mission()
    
    async def abort_mission(self) -> bool:
        """Aborts the mission"""
        return await self.mission_manager.abort_current_mission()
    
    def add_custom_mission(self, mission: Mission):
        """Adds a custom mission"""
        self.mission_manager.register_mission(mission)