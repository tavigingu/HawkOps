import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from datatypes import MissionResult, MissionStatus
from drone_controller import ParrotAnafi

if TYPE_CHECKING:
    from drone_controller import ParrotAnafi

logger = logging.getLogger(__name__)

class Mission(ABC):
    """Clasa abstractă pentru toate misiunile dronei"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = MissionStatus.IDLE
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self._is_paused = False
    
    @abstractmethod
    async def execute(self, drone: 'ParrotAnafi') -> MissionResult:
        """Execută misiunea"""
        pass
    
    async def pause(self) -> bool:
        """Pune misiunea în pauză"""
        if self.status == MissionStatus.EXECUTING:
            self.status = MissionStatus.PAUSED
            self._is_paused = True
            logger.info(f"Misiunea {self.name} pusă în pauză")
            return True
        return False
    
    async def resume(self) -> bool:
        """Reia misiunea din pauză"""
        if self.status == MissionStatus.PAUSED:
            self.status = MissionStatus.EXECUTING
            self._is_paused = False
            logger.info(f"Misiunea {self.name} reluată")
            return True
        return False
    
    async def abort(self) -> bool:
        """Anulează misiunea"""
        if self.status in [MissionStatus.EXECUTING, MissionStatus.PAUSED]:
            self.status = MissionStatus.FAILED
            logger.info(f"Misiunea {self.name} anulată")
            return True
        return False
    
    def get_progress(self) -> float:
        """Returnează progresul misiunii (0.0 - 1.0)"""
        return self.progress
    
    def get_status(self) -> MissionStatus:
        """Returnează statusul curent al misiunii"""
        return self.status
    
    async def _wait_if_paused(self):
        """Utility pentru a aștepta când misiunea este în pauză"""
        while self._is_paused:
            await asyncio.sleep(0.1)
