from typing import List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, name: str):
        """
        Initialize a memory object.
        :param name: Name of the memory type.
        """
        self.name = name
        self.data: List[Any] = []

    def store(self, data: Any) -> None:
        """
        Store data in memory.
        :param data: Data to store.
        """
        self.data.append(data)
        logger.debug(f"{self.name} Memory: Data stored.")

    def retrieve(self, query: Optional[Any] = None) -> List[Any]:
        """
        Retrieve data from memory.
        :param query: Query to filter data.
        """
        if query is None:
            return self.data
        return [item for item in self.data if query in str(item)]

    def clear(self) -> None:
        """
        Clear all data from memory.
        """
        self.data = []
        logger.debug(f"{self.name} Memory: Data cleared.")
