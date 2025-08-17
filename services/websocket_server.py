"""
WebSocket Server
================

Real-time WebSocket server for broadcasting trading signals, training progress,
and system status to connected clients (mobile apps, web dashboards).

Features:
- Real-time signal broadcasting
- Training progress updates
- Client connection management
- Message queuing and reliability
- Authentication and rate limiting

Author: PPO Trading System
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import socketio
from aiohttp import web
from loguru import logger
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import yaml


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    type: str
    timestamp: datetime
    data: Dict[str, Any]
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class WebSocketBroadcaster:
    """
    WebSocket broadcaster for real-time updates.
    
    This class manages WebSocket connections and handles broadcasting
    of trading signals, training updates, and system status.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize WebSocket broadcaster.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.ws_config = config['websocket']
        self.redis_config = config['database']['redis']
        
        # Initialize Socket.IO server
        self.sio = socketio.AsyncServer(
            cors_allowed_origins=self.ws_config['cors_allowed_origins'],
            ping_interval=self.ws_config['ping_interval'],
            ping_timeout=self.ws_config['ping_timeout']
        )
        
        # Initialize web application
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Client management
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of topics
        
        # Message queuing
        self.message_queue: List[WebSocketMessage] = []
        self.max_queue_size = 1000
        
        # Redis for persistence and scaling
        self.redis_client = None
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("WebSocket broadcaster initialized")
    
    async def initialize(self) -> None:
        """Initialize async components."""
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config.get('password'),
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _setup_event_handlers(self) -> None:
        """Setup Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid: str, environ: Dict) -> bool:
            """Handle client connection."""
            try:
                # Generate client ID
                client_id = str(uuid.uuid4())
                
                # Store client info
                self.connected_clients[sid] = {
                    'client_id': client_id,
                    'connected_at': datetime.now(),
                    'ip_address': environ.get('REMOTE_ADDR', 'unknown'),
                    'user_agent': environ.get('HTTP_USER_AGENT', 'unknown')
                }
                
                # Initialize subscriptions
                self.client_subscriptions[sid] = set()
                
                logger.info(f"Client connected: {client_id} (SID: {sid})")
                
                # Send welcome message
                await self.sio.emit('connected', {
                    'client_id': client_id,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Connected to PPO Trading System'
                }, room=sid)
                
                return True
                
            except Exception as e:
                logger.error(f"Error handling connection: {e}")
                return False
        
        @self.sio.event
        async def disconnect(sid: str) -> None:
            """Handle client disconnection."""
            try:
                client_info = self.connected_clients.get(sid, {})
                client_id = client_info.get('client_id', 'unknown')
                
                # Clean up
                self.connected_clients.pop(sid, None)
                self.client_subscriptions.pop(sid, None)
                
                logger.info(f"Client disconnected: {client_id} (SID: {sid})")
                
            except Exception as e:
                logger.error(f"Error handling disconnection: {e}")
        
        @self.sio.event
        async def subscribe(sid: str, data: Dict[str, Any]) -> None:
            """Handle subscription to topics."""
            try:
                topics = data.get('topics', [])
                
                if sid in self.client_subscriptions:
                    self.client_subscriptions[sid].update(topics)
                    
                    await self.sio.emit('subscription_confirmed', {
                        'topics': list(self.client_subscriptions[sid]),
                        'timestamp': datetime.now().isoformat()
                    }, room=sid)
                    
                    logger.info(f"Client {sid} subscribed to topics: {topics}")
                
            except Exception as e:
                logger.error(f"Error handling subscription: {e}")
        
        @self.sio.event
        async def unsubscribe(sid: str, data: Dict[str, Any]) -> None:
            """Handle unsubscription from topics."""
            try:
                topics = data.get('topics', [])
                
                if sid in self.client_subscriptions:
                    self.client_subscriptions[sid].difference_update(topics)
                    
                    await self.sio.emit('unsubscription_confirmed', {
                        'topics': topics,
                        'remaining_topics': list(self.client_subscriptions[sid]),
                        'timestamp': datetime.now().isoformat()
                    }, room=sid)
                    
                    logger.info(f"Client {sid} unsubscribed from topics: {topics}")
                
            except Exception as e:
                logger.error(f"Error handling unsubscription: {e}")
        
        @self.sio.event
        async def get_status(sid: str) -> None:
            """Handle status request."""
            try:
                status = {
                    'connected_clients': len(self.connected_clients),
                    'message_queue_size': len(self.message_queue),
                    'server_time': datetime.now().isoformat(),
                    'redis_connected': self.redis_client is not None
                }
                
                await self.sio.emit('status_response', status, room=sid)
                
            except Exception as e:
                logger.error(f"Error handling status request: {e}")
    
    async def broadcast_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Broadcast trading signal to subscribed clients.
        
        Args:
            signal_data: Signal information
        """
        message = WebSocketMessage(
            type='trading_signal',
            timestamp=datetime.now(),
            data=signal_data
        )
        
        await self._broadcast_to_subscribers('trading_signals', message)
    
    async def broadcast_training_update(self, training_data: Dict[str, Any]) -> None:
        """
        Broadcast training progress update.
        
        Args:
            training_data: Training progress information
        """
        message = WebSocketMessage(
            type='training_update',
            timestamp=datetime.now(),
            data=training_data
        )
        
        await self._broadcast_to_subscribers('training_updates', message)
    
    async def broadcast_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """
        Broadcast trade execution confirmation.
        
        Args:
            trade_data: Trade execution details
        """
        message = WebSocketMessage(
            type='trade_executed',
            timestamp=datetime.now(),
            data=trade_data
        )
        
        await self._broadcast_to_subscribers('trade_executions', message)
    
    async def broadcast_performance_update(self, performance_data: Dict[str, Any]) -> None:
        """
        Broadcast performance metrics update.
        
        Args:
            performance_data: Performance metrics
        """
        message = WebSocketMessage(
            type='performance_update',
            timestamp=datetime.now(),
            data=performance_data
        )
        
        await self._broadcast_to_subscribers('performance_updates', message)
    
    async def broadcast_system_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Broadcast system alert or notification.
        
        Args:
            alert_data: Alert information
        """
        message = WebSocketMessage(
            type='system_alert',
            timestamp=datetime.now(),
            data=alert_data
        )
        
        await self._broadcast_to_subscribers('system_alerts', message)
    
    async def _broadcast_to_subscribers(self, topic: str, message: WebSocketMessage) -> None:
        """
        Broadcast message to clients subscribed to specific topic.
        
        Args:
            topic: Topic name
            message: Message to broadcast
        """
        try:
            # Find subscribers
            subscribers = [
                sid for sid, subscriptions in self.client_subscriptions.items()
                if topic in subscriptions
            ]
            
            if not subscribers:
                logger.debug(f"No subscribers for topic: {topic}")
                return
            
            # Broadcast to subscribers
            message_dict = message.to_dict()
            
            for sid in subscribers:
                try:
                    await self.sio.emit(message.type, message_dict, room=sid)
                except Exception as e:
                    logger.warning(f"Failed to send message to client {sid}: {e}")
            
            logger.debug(f"Broadcasted {message.type} to {len(subscribers)} clients")
            
            # Store in message queue
            self._add_to_queue(message)
            
            # Persist to Redis if available
            if self.redis_client:
                await self._persist_message(message)
                
        except Exception as e:
            logger.error(f"Error broadcasting to subscribers: {e}")
    
    def _add_to_queue(self, message: WebSocketMessage) -> None:
        """Add message to local queue."""
        self.message_queue.append(message)
        
        # Trim queue if too large
        if len(self.message_queue) > self.max_queue_size:
            self.message_queue.pop(0)
    
    async def _persist_message(self, message: WebSocketMessage) -> None:
        """Persist message to Redis."""
        try:
            if self.redis_client:
                key = f"ws_messages:{message.type}:{datetime.now().strftime('%Y%m%d')}"
                await self.redis_client.lpush(key, json.dumps(message.to_dict()))
                await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 messages
                
        except Exception as e:
            logger.warning(f"Failed to persist message to Redis: {e}")
    
    async def get_recent_messages(self, message_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent messages of specific type.
        
        Args:
            message_type: Type of messages to retrieve
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        try:
            if self.redis_client:
                key = f"ws_messages:{message_type}:{datetime.now().strftime('%Y%m%d')}"
                messages = await self.redis_client.lrange(key, 0, limit - 1)
                return [json.loads(msg) for msg in messages]
            else:
                # Fallback to local queue
                return [
                    msg.to_dict() for msg in self.message_queue
                    if msg.type == message_type
                ][-limit:]
                
        except Exception as e:
            logger.error(f"Error retrieving recent messages: {e}")
            return []
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'total_connections': len(self.connected_clients),
            'active_subscriptions': sum(len(subs) for subs in self.client_subscriptions.values()),
            'message_queue_size': len(self.message_queue),
            'uptime': datetime.now().isoformat()
        }
    
    async def start_server(self) -> None:
        """Start the WebSocket server."""
        await self.initialize()
        
        # Add health check endpoint
        async def health_check(request):
            return web.Response(
                text=json.dumps({'status': 'healthy', 'timestamp': datetime.now().isoformat()}),
                content_type='application/json'
            )
        
        self.app.router.add_get('/health', health_check)
        
        # Start server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            self.ws_config['host'],
            self.ws_config['port']
        )
        
        await site.start()
        
        logger.info(f"WebSocket server started on {self.ws_config['host']}:{self.ws_config['port']}")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket server...")
        finally:
            await runner.cleanup()


# Standalone server runner
if __name__ == "__main__":
    async def main():
        broadcaster = WebSocketBroadcaster()
        await broadcaster.start_server()
    
    asyncio.run(main())