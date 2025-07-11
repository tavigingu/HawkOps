import asyncio
import logging
from typing import List
from datatypes import MissionResult, MissionStatus, Position
from drone_controller import ParrotAnafi
from mission_base import Mission

logger = logging.getLogger(__name__)

class RoomSearchMission(Mission):
    """Main mission: search rooms and detect people"""
    
    def __init__(self, rooms: List[str], search_altitude: float = 2.0):
        super().__init__("Room Search", "Search rooms and detect people")
        self.rooms = rooms
        self.search_altitude = search_altitude
        self.detected_persons = []
        self.current_room_index = 0
    
    async def execute(self, drone: 'ParrotAnafi') -> MissionResult:
        self.status = MissionStatus.EXECUTING
        logger.info(f"Starting search mission in {len(self.rooms)} rooms")
        
        try:
            for i, room in enumerate(self.rooms):
                await self._wait_if_paused()
                
                if self.status == MissionStatus.FAILED:
                    break
                
                self.current_room_index = i
                self.progress = i / len(self.rooms)
                
                logger.info(f"Searching in {room} ({i+1}/{len(self.rooms)})")
                await self._search_room(drone, room)
            
            self.status = MissionStatus.COMPLETED
            self.progress = 1.0
            
            return MissionResult(
                success=True,
                message=f"Mission completed. {len(self.detected_persons)} persons detected",
                data={
                    "detected_persons": self.detected_persons,
                    "rooms_searched": len(self.rooms)
                }
            )
            
        except Exception as e:
            self.status = MissionStatus.FAILED
            logger.error(f"Error in Room Search: {e}")
            return MissionResult(success=False, message=f"Mission error: {str(e)}")
    
    async def _search_room(self, drone: 'ParrotAnafi', room: str):
        """Perform search in a specific room"""
        # Move to search altitude
        await drone.move_to_altitude(self.search_altitude)
        
        # Simulate a search pattern
        await self._perform_search_pattern(drone)
        
        # Simulate detection (to be replaced with real logic)
        await self._simulate_person_detection(room)
    
    async def _perform_search_pattern(self, drone: 'ParrotAnafi'):
        """Perform a search pattern inside the room"""
        # Simple pattern: zigzag movement
        search_points = [
            Position(0, 0, self.search_altitude),
            Position(3, 0, self.search_altitude),
            Position(3, 3, self.search_altitude),
            Position(0, 3, self.search_altitude)
        ]
        
        for point in search_points:
            await self._wait_if_paused()
            await drone.move_to_position(point)
            await asyncio.sleep(1)  # Time to analyze surroundings
    
    async def _simulate_person_detection(self, room: str):
        """Simulate person detection"""
        # Simulation – will be replaced with real detection logic
        import random
        if random.random() > 0.7:  # 30% chance of detection
            self.detected_persons.append({
                "room": room,
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": random.uniform(0.8, 1.0)
            })
            logger.info(f"Person detected in {room}")
