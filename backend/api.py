"""
Backend API
===========

FastAPI-based REST API for the PPO Trading System.
Provides endpoints for system monitoring, model management, and trading control.

Endpoints:
- System status and health checks
- Model management (list, load, train)
- Trading control (start/stop live trading)
- Performance metrics and analytics
- WebSocket connection info

Author: PPO Trading System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timedelta
import uuid
import os
from loguru import logger
import yaml

from backend.data_manager import DataManager
from services.websocket_server import WebSocketBroadcaster
from models.ppo_agent import PPOTradingAgent
from services.mt5_client import MT5Client
from services.signal_generator import SignalGenerator


# Pydantic models for API requests/responses
class SystemStatus(BaseModel):
    """System status response model."""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Current timestamp")
    uptime: str = Field(..., description="System uptime")
    components: Dict[str, str] = Field(..., description="Component statuses")
    metrics: Dict[str, Any] = Field(..., description="System metrics")


class TrainingRequest(BaseModel):
    """Training request model."""
    symbol: str = Field(..., description="Trading symbol")
    start_date: Optional[str] = Field(None, description="Start date for training data")
    end_date: Optional[str] = Field(None, description="End date for training data")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")
    session_name: Optional[str] = Field(None, description="Training session name")


class TradingControlRequest(BaseModel):
    """Trading control request model."""
    action: str = Field(..., description="Action: start or stop")
    symbols: Optional[List[str]] = Field(None, description="Symbols to trade")
    demo_mode: bool = Field(True, description="Use demo trading")


class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model file path")
    created_at: datetime = Field(..., description="Creation timestamp")
    size_mb: float = Field(..., description="File size in MB")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")


class TradingSystemAPI:
    """Main API class for the PPO Trading System."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the API."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_config = self.config['api']
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=self.api_config['title'],
            description=self.api_config['description'],
            version=self.api_config['version']
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on your needs
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.data_manager = DataManager()
        self.websocket_broadcaster = None  # Will be initialized in startup
        self.ppo_agent = None
        self.mt5_client = None
        self.signal_generator = None
        
        # System state
        self.system_start_time = datetime.now()
        self.is_training = False
        self.is_live_trading = False
        self.current_training_session = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Trading System API initialized")
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize components on startup."""
            try:
                # Initialize data manager
                await self.data_manager.initialize()
                
                # Initialize WebSocket broadcaster
                self.websocket_broadcaster = WebSocketBroadcaster()
                await self.websocket_broadcaster.initialize()
                
                # Initialize other components
                self.ppo_agent = PPOTradingAgent(websocket_broadcaster=self.websocket_broadcaster)
                self.mt5_client = MT5Client()
                self.signal_generator = SignalGenerator()
                
                logger.info("All components initialized successfully")
                
            except Exception as e:
                logger.error(f"Error during startup: {e}")
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            try:
                if self.mt5_client:
                    self.mt5_client.disconnect()
                
                if self.data_manager:
                    await self.data_manager.close()
                
                logger.info("System shutdown completed")
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # System status endpoint
        @self.app.get("/api/status", response_model=SystemStatus)
        async def get_system_status():
            """Get comprehensive system status."""
            uptime = datetime.now() - self.system_start_time
            
            # Check component statuses
            components = {
                "database": "connected" if self.data_manager.is_connected() else "disconnected",
                "websocket": "active" if self.websocket_broadcaster else "inactive",
                "mt5": "connected" if self.mt5_client and self.mt5_client.is_connected else "disconnected",
                "model": "loaded" if self.ppo_agent and self.ppo_agent.model else "not_loaded"
            }
            
            # System metrics
            metrics = {
                "training_active": self.is_training,
                "live_trading_active": self.is_live_trading,
                "current_session": self.current_training_session,
                "uptime_seconds": uptime.total_seconds()
            }
            
            # Add WebSocket metrics if available
            if self.websocket_broadcaster:
                ws_stats = await self.websocket_broadcaster.get_connection_stats()
                metrics.update(ws_stats)
            
            return SystemStatus(
                status="operational" if all(status != "error" for status in components.values()) else "degraded",
                timestamp=datetime.now(),
                uptime=str(uptime),
                components=components,
                metrics=metrics
            )
        
        # Model management endpoints
        @self.app.get("/api/models", response_model=List[ModelInfo])
        async def list_models():
            """List available trained models."""
            try:
                models_dir = "./models/saved_models"
                if not os.path.exists(models_dir):
                    return []
                
                models = []
                for filename in os.listdir(models_dir):
                    if filename.endswith(".zip"):
                        filepath = os.path.join(models_dir, filename)
                        stat = os.stat(filepath)
                        
                        # Get performance metrics if available
                        metrics = await self.data_manager.get_model_performance(filename)
                        
                        models.append(ModelInfo(
                            name=filename,
                            path=filepath,
                            created_at=datetime.fromtimestamp(stat.st_mtime),
                            size_mb=stat.st_size / (1024 * 1024),
                            performance_metrics=metrics
                        ))
                
                return sorted(models, key=lambda x: x.created_at, reverse=True)
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail="Failed to list models")
        
        @self.app.post("/api/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load a specific model for inference."""
            try:
                model_path = f"./models/saved_models/{model_name}"
                if not os.path.exists(model_path):
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if not self.ppo_agent:
                    self.ppo_agent = PPOTradingAgent(websocket_broadcaster=self.websocket_broadcaster)
                
                success = self.ppo_agent.load_model(model_path)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load model")
                
                return {"message": f"Model {model_name} loaded successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Training endpoints
        @self.app.post("/api/training/start")
        async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
            """Start model training."""
            if self.is_training:
                raise HTTPException(status_code=400, detail="Training already in progress")
            
            try:
                # Generate session ID
                session_id = str(uuid.uuid4())
                self.current_training_session = session_id
                
                # Add training task to background
                background_tasks.add_task(
                    self._run_training,
                    session_id,
                    request
                )
                
                return {
                    "message": "Training started",
                    "session_id": session_id
                }
                
            except Exception as e:
                logger.error(f"Error starting training: {e}")
                raise HTTPException(status_code=500, detail="Failed to start training")
        
        @self.app.post("/api/training/stop")
        async def stop_training():
            """Stop current training session."""
            if not self.is_training:
                raise HTTPException(status_code=400, detail="No training in progress")
            
            # Implementation depends on how training cancellation is handled
            self.is_training = False
            self.current_training_session = None
            
            return {"message": "Training stop requested"}
        
        @self.app.get("/api/training/status")
        async def get_training_status():
            """Get current training status."""
            return {
                "is_training": self.is_training,
                "session_id": self.current_training_session,
                "start_time": self.system_start_time.isoformat() if self.is_training else None
            }
        
        # Trading control endpoints
        @self.app.post("/api/trading/control")
        async def control_trading(request: TradingControlRequest):
            """Start or stop live trading."""
            try:
                if request.action == "start":
                    if self.is_live_trading:
                        raise HTTPException(status_code=400, detail="Trading already active")
                    
                    # Validate prerequisites
                    if not self.ppo_agent or not self.ppo_agent.model:
                        raise HTTPException(status_code=400, detail="No model loaded")
                    
                    # Connect to MT5 if not connected
                    if not self.mt5_client.is_connected:
                        if not self.mt5_client.connect():
                            raise HTTPException(status_code=500, detail="Failed to connect to MT5")
                    
                    # Start live trading
                    await self._start_live_trading(request.symbols or ["XAUUSD"], request.demo_mode)
                    
                    return {"message": "Live trading started"}
                
                elif request.action == "stop":
                    if not self.is_live_trading:
                        raise HTTPException(status_code=400, detail="Trading not active")
                    
                    await self._stop_live_trading()
                    
                    return {"message": "Live trading stopped"}
                
                else:
                    raise HTTPException(status_code=400, detail="Invalid action")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error controlling trading: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Performance and analytics endpoints
        @self.app.get("/api/performance/{session_id}")
        async def get_performance_metrics(session_id: str):
            """Get performance metrics for a training session."""
            try:
                metrics = await self.data_manager.get_training_metrics(session_id)
                if not metrics:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                return metrics
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/signals/recent")
        async def get_recent_signals(limit: int = 50):
            """Get recent trading signals."""
            try:
                if self.websocket_broadcaster:
                    signals = await self.websocket_broadcaster.get_recent_messages("trading_signal", limit)
                    return signals
                else:
                    return []
                    
            except Exception as e:
                logger.error(f"Error getting recent signals: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # WebSocket info endpoint
        @self.app.get("/api/websocket/stats")
        async def get_websocket_stats():
            """Get WebSocket connection statistics."""
            if not self.websocket_broadcaster:
                raise HTTPException(status_code=503, detail="WebSocket server not available")
            
            try:
                stats = await self.websocket_broadcaster.get_connection_stats()
                return stats
                
            except Exception as e:
                logger.error(f"Error getting WebSocket stats: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _run_training(self, session_id: str, request: TrainingRequest) -> None:
        """Run training in background."""
        self.is_training = True
        
        try:
            logger.info(f"Starting training session {session_id}")
            
            # Get training data
            train_data = await self.data_manager.get_historical_data(
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            if train_data.empty:
                logger.error("No training data available")
                return
            
            # Create and train model
            if not self.ppo_agent:
                self.ppo_agent = PPOTradingAgent(websocket_broadcaster=self.websocket_broadcaster)
            
            results = self.ppo_agent.train(
                train_data=train_data,
                session_id=session_id
            )
            
            # Save training results
            await self.data_manager.save_training_session(session_id, results)
            
            logger.info(f"Training session {session_id} completed")
            
        except Exception as e:
            logger.error(f"Error in training session {session_id}: {e}")
        finally:
            self.is_training = False
            self.current_training_session = None
    
    async def _start_live_trading(self, symbols: List[str], demo_mode: bool) -> None:
        """Start live trading."""
        self.is_live_trading = True
        
        # Implementation for live trading startup
        logger.info(f"Live trading started for symbols: {symbols}, demo_mode: {demo_mode}")
    
    async def _stop_live_trading(self) -> None:
        """Stop live trading."""
        self.is_live_trading = False
        
        # Implementation for live trading shutdown
        logger.info("Live trading stopped")


# Create API instance
api = TradingSystemAPI()
app = api.app

# Run with: uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)