"""PostgreSQL-backed chat message history for LangChain.

Follows the ``langchain-postgres`` pattern: a single table with
``session_id``, ``message`` (JSONB), and ``created_at`` columns.
No Apache AGE or pgvector dependency — works with any PostgreSQL instance.

Supports both synchronous and asynchronous access via ``psycopg3``.

Usage (sync)::

    from langchain_age import PostgresChatMessageHistory

    history = PostgresChatMessageHistory.create(
        connection_string="host=localhost dbname=mydb user=foo password=bar",
        session_id="user-123",
    )
    history.add_user_message("Hello")
    history.add_ai_message("Hi there!")
    print(history.messages)

Usage (async)::

    history = await PostgresChatMessageHistory.acreate(
        connection_string="host=localhost dbname=mydb user=foo password=bar",
        session_id="user-123",
    )
    await history.aadd_messages([HumanMessage(content="Hello")])
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import psycopg
    import psycopg.rows
except ImportError as e:
    raise ImportError(
        "psycopg is required for PostgresChatMessageHistory.\n"
        "Install: pip install 'psycopg[binary]'"
    ) from e

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from langchain_age.utils.cypher import validate_sql_identifier

_DEFAULT_TABLE = "langchain_age_chat_history"


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a PostgreSQL table.

    Compatible with any PostgreSQL instance — does not require Apache AGE or
    pgvector.  Mirrors ``PostgresChatMessageHistory`` from *langchain-postgres*.

    The table schema::

        CREATE TABLE IF NOT EXISTS "<table_name>" (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL,
            message     JSONB NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT now()
        );

    Args:
        table_name: SQL table name (validated at construction).
        session_id: Unique identifier for this conversation thread.
        sync_connection: A ``psycopg.Connection`` instance.  Either this or
            ``async_connection`` must be provided.
        async_connection: A ``psycopg.AsyncConnection`` instance.
    """

    def __init__(
        self,
        table_name: str,
        session_id: str,
        *,
        sync_connection: Optional[psycopg.Connection] = None,
        async_connection: Optional["psycopg.AsyncConnection"] = None,
    ) -> None:
        validate_sql_identifier(table_name, context="table_name")
        if sync_connection is None and async_connection is None:
            raise ValueError(
                "At least one of sync_connection or async_connection is required."
            )
        self._table = table_name
        self._session_id = session_id
        self._conn = sync_connection
        self._aconn = async_connection

    # ------------------------------------------------------------------
    # Factory helpers (mirrors langchain-postgres pattern)
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        connection_string: str,
        session_id: str,
        table_name: str = _DEFAULT_TABLE,
    ) -> PostgresChatMessageHistory:
        """Create a sync instance and ensure the table exists."""
        conn = psycopg.connect(connection_string, autocommit=True)
        inst = cls(table_name=table_name, session_id=session_id, sync_connection=conn)
        inst.create_tables()
        return inst

    @classmethod
    async def acreate(
        cls,
        connection_string: str,
        session_id: str,
        table_name: str = _DEFAULT_TABLE,
    ) -> PostgresChatMessageHistory:
        """Create an async instance and ensure the table exists."""
        conn = await psycopg.AsyncConnection.connect(
            connection_string, autocommit=True
        )
        inst = cls(
            table_name=table_name, session_id=session_id, async_connection=conn
        )
        await inst.acreate_tables()
        return inst

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create the chat history table if it does not exist (sync)."""
        assert self._conn is not None
        tbl = f'"{self._table}"'
        idx = f'"{self._table}_session_idx"'
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                message     JSONB NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
        """)
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} (session_id);"
        )

    async def acreate_tables(self) -> None:
        """Create the chat history table if it does not exist (async)."""
        assert self._aconn is not None
        tbl = f'"{self._table}"'
        idx = f'"{self._table}_session_idx"'
        await self._aconn.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                message     JSONB NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
        """)
        await self._aconn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} (session_id);"
        )

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages for this session (sync)."""
        return self.get_messages()

    def get_messages(self) -> List[BaseMessage]:
        """Fetch messages ordered by creation time (sync)."""
        assert self._conn is not None
        tbl = f'"{self._table}"'
        rows = self._conn.execute(
            f"SELECT message FROM {tbl} WHERE session_id = %s ORDER BY id;",
            (self._session_id,),
        ).fetchall()
        return messages_from_dict([row[0] for row in rows])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages at once (sync)."""
        assert self._conn is not None
        tbl = f'"{self._table}"'
        params = [
            (self._session_id, psycopg.types.json.Jsonb(self._msg_to_dict(m)))
            for m in messages
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {tbl} (session_id, message) VALUES (%s, %s);",
                params,
            )

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message (sync)."""
        self.add_messages([message])

    def clear(self) -> None:
        """Delete all messages for this session (sync)."""
        assert self._conn is not None
        tbl = f'"{self._table}"'
        self._conn.execute(
            f"DELETE FROM {tbl} WHERE session_id = %s;",
            (self._session_id,),
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aget_messages(self) -> List[BaseMessage]:
        """Fetch messages (async)."""
        assert self._aconn is not None
        tbl = f'"{self._table}"'
        cur = await self._aconn.execute(
            f"SELECT message FROM {tbl} WHERE session_id = %s ORDER BY id;",
            (self._session_id,),
        )
        rows = await cur.fetchall()
        return messages_from_dict([row[0] for row in rows])

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages (async)."""
        assert self._aconn is not None
        tbl = f'"{self._table}"'
        params = [
            (self._session_id, psycopg.types.json.Jsonb(self._msg_to_dict(m)))
            for m in messages
        ]
        async with self._aconn.cursor() as cur:
            for p in params:
                await cur.execute(
                    f"INSERT INTO {tbl} (session_id, message) VALUES (%s, %s);",
                    p,
                )

    async def aclear(self) -> None:
        """Delete all messages for this session (async)."""
        assert self._aconn is not None
        tbl = f'"{self._table}"'
        await self._aconn.execute(
            f"DELETE FROM {tbl} WHERE session_id = %s;",
            (self._session_id,),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_to_dict(message: BaseMessage) -> dict:
        """Serialise a LangChain message to a JSON-safe dict."""
        return {"type": message.type, "data": {"content": message.content, **message.additional_kwargs}}

    def close(self) -> None:
        """Close underlying connections."""
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"PostgresChatMessageHistory(table='{self._table}', "
            f"session='{self._session_id}')"
        )
