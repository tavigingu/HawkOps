import logging
from typing import Dict, List, Optional
from mission_base import Mission
from datatypes import MissionResult
from drone_controller import ParrotAnafi

logger = logging.getLogger(__name__)

class MissionManager:
    """Manager pentru gestionarea misiunilor"""
    
    def __init__(self):
        self.missions: Dict[str, Mission] = {}
        self.current_mission: Optional[Mission] = None
        self.mission_history: List[Mission] = []
    
    def register_mission(self, mission: Mission):
        """Înregistrează o misiune nouă"""
        self.missions[mission.name] = mission
        logger.info(f"Misiune înregistrată: {mission.name}")
    
    def get_available_missions(self) -> List[str]:
        """Returnează lista misiunilor disponibile"""
        return list(self.missions.keys())
    
    def get_mission_info(self, mission_name: str) -> Optional[Dict]:
        """Returnează informații despre o misiune"""
        if mission_name in self.missions:
            mission = self.missions[mission_name]
            return {
                "name": mission.name,
                "description": mission.description,
                "status": mission.status.value,
                "progress": mission.progress
            }
        return None
    
    async def execute_mission(self, mission_name: str, drone: 'ParrotAnafi') -> MissionResult:
        """Execută o misiune specifică"""
        if mission_name not in self.missions:
            return MissionResult(success=False, message=f"Misiunea '{mission_name}' nu există")
        
        mission = self.missions[mission_name]
        self.current_mission = mission
        
        try:
            logger.info(f"Începe execuția misiunii: {mission_name}")
            result = await mission.execute(drone)
            self.mission_history.append(mission)
            return result
        except Exception as e:
            logger.error(f"Eroare în execuția misiunii {mission_name}: {e}")
            return MissionResult(success=False, message=f"Eroare: {str(e)}")
        finally:
            self.current_mission = None
    
    async def pause_current_mission(self) -> bool:
        """Pune misiunea curentă în pauză"""
        if self.current_mission:
            return await self.current_mission.pause()
        return False
    
    async def resume_current_mission(self) -> bool:
        """Reia misiunea curentă"""
        if self.current_mission:
            return await self.current_mission.resume()
        return False
    
    async def abort_current_mission(self) -> bool:
        """Anulează misiunea curentă"""
        if self.current_mission:
            return await self.current_mission.abort()
        return False
    
    def get_current_mission_status(self) -> Optional[Dict]:
        """Returnează statusul misiunii curente"""
        if self.current_mission:
            return {
                "name": self.current_mission.name,
                "status": self.current_mission.status.value,
                "progress": self.current_mission.progress
            }
        return None