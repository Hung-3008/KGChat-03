from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DatabaseClient(ABC):
    """
    Abstract base class cho tất cả database clients trong hệ thống knowledge graph.

    Định nghĩa interface chung cho việc kết nối, import/export data, và thao tác với database.
    """

    def __init__(self, **kwargs):
        """Initialize database client với các parameters chung."""
        self._connection = None
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """
        Kết nối tới database.

        Returns:
            True nếu kết nối thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Đóng kết nối database."""
        pass

    @abstractmethod
    async def verify_connectivity(self) -> bool:
        """
        Kiểm tra kết nối database có hoạt động không.

        Returns:
            True nếu kết nối OK, False nếu có lỗi
        """
        pass

    @abstractmethod
    async def setup_schema(self) -> bool:
        """
        Thiết lập schema/structure cho database.

        Returns:
            True nếu setup thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def clear_database(self) -> bool:
        """
        Xóa toàn bộ dữ liệu trong database.

        WARNING: Thao tác này xóa tất cả dữ liệu!

        Returns:
            True nếu xóa thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def import_data(self, data: Any, **kwargs) -> bool:
        """
        Import dữ liệu vào database.

        Args:
            data: Dữ liệu cần import (format tùy thuộc implementation)
            **kwargs: Các tham số bổ sung

        Returns:
            True nếu import thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Thực hiện query/search trong database.

        Args:
            query_params: Tham số query (format tùy thuộc implementation)

        Returns:
            Danh sách kết quả query
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về database.

        Returns:
            Dictionary chứa các thông tin thống kê
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check xem database có đang kết nối không."""
        return self._is_connected

    async def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra tình trạng sức khỏe của database connection.

        Returns:
            Dictionary chứa thông tin health check
        """
        connectivity = await self.verify_connectivity()
        stats = await self.get_statistics() if connectivity else {}

        return {
            "connected": connectivity,
            "statistics": stats,
            "client_type": self.__class__.__name__
        }


class GraphDatabaseClient(DatabaseClient):
    """
    Abstract base class cho graph database clients (Neo4j, etc.).
    """

    @abstractmethod
    async def import_nodes(self, nodes: List[Dict[str, Any]], **kwargs) -> bool:
        """
        Import nodes vào graph database.

        Args:
            nodes: Danh sách nodes cần import
            **kwargs: Các tham số bổ sung (label, batch_size, etc.)

        Returns:
            True nếu import thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def import_relationships(self, relationships: List[Dict[str, Any]], **kwargs) -> bool:
        """
        Import relationships vào graph database.

        Args:
            relationships: Danh sách relationships cần import
            **kwargs: Các tham số bổ sung (source_label, target_label, etc.)

        Returns:
            True nếu import thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Thực hiện Cypher query (hoặc graph query tương tự).

        Args:
            query: Graph query string
            params: Tham số cho query

        Returns:
            Danh sách kết quả query
        """
        pass


class VectorDatabaseClient(DatabaseClient):
    """
    Abstract base class cho vector database clients (Qdrant, Pinecone, etc.).
    """

    @abstractmethod
    async def create_collections(self, collection_configs: List[Dict[str, Any]]) -> bool:
        """
        Tạo collections/indexes cho vector database.

        Args:
            collection_configs: Danh sách config cho các collections

        Returns:
            True nếu tạo thành công, False nếu thất bại
        """
        pass

    @abstractmethod
    async def store_vectors(self, vectors: List[Dict[str, Any]], collection_name: str, **kwargs) -> int:
        """
        Lưu trữ vectors vào database.

        Args:
            vectors: Danh sách vectors và metadata
            collection_name: Tên collection để lưu
            **kwargs: Các tham số bổ sung

        Returns:
            Số lượng vectors đã lưu thành công
        """
        pass

    @abstractmethod
    async def search_vectors(self, query_vector: List[float], collection_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Tìm kiếm vectors tương tự.

        Args:
            query_vector: Vector để tìm kiếm
            collection_name: Tên collection để search
            **kwargs: Các tham số bổ sung (limit, score_threshold, etc.)

        Returns:
            Danh sách kết quả tìm kiếm với điểm số
        """
        pass

    @abstractmethod
    async def retrieve_by_ids(self, ids: List[str], collection_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Lấy vectors theo IDs.

        Args:
            ids: Danh sách IDs cần lấy
            collection_name: Tên collection
            **kwargs: Các tham số bổ sung

        Returns:
            Danh sách vectors và metadata
        """
        pass
