"""
Security Monitoring and Incident Response System
Real-time threat detection and automated response
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import hashlib
import re

import redis
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import aiohttp
from dataclasses import dataclass, asdict


# Metrics
security_events = Counter(
    'security_events_total',
    'Total security events detected',
    ['event_type', 'severity', 'action_taken']
)

threat_score = Gauge(
    'threat_score_current',
    'Current threat score for IP/User',
    ['identifier_type', 'identifier']
)

blocked_requests = Counter(
    'blocked_requests_total',
    'Total blocked requests',
    ['reason', 'ip_address']
)

incident_response_time = Histogram(
    'incident_response_seconds',
    'Time to respond to security incidents',
    ['incident_type']
)


@dataclass
class SecurityEvent:
    """Security event data structure"""
    timestamp: datetime
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    raw_request: Optional[str] = None
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ThreatIndicator:
    """Threat indicator definition"""
    pattern: str
    description: str
    severity: str
    category: str
    
    def matches(self, text: str) -> bool:
        """Check if indicator matches text"""
        return bool(re.search(self.pattern, text, re.IGNORECASE))


class ThreatIntelligence:
    """Threat intelligence and pattern matching"""
    
    def __init__(self):
        self.indicators = self._load_threat_indicators()
        self.logger = structlog.get_logger()
    
    def _load_threat_indicators(self) -> List[ThreatIndicator]:
        """Load threat indicators"""
        return [
            # SQL Injection
            ThreatIndicator(
                r"(union|select|insert|update|delete|drop|create)\s+(from|into|table|database)",
                "SQL injection attempt",
                "high",
                "sql_injection"
            ),
            ThreatIndicator(
                r"(';|--;|/\*|\*/|@@|@|char|nchar|varchar|nvarchar|alter|begin|cast|create|cursor|declare|delete|drop|end|exec|execute|fetch|insert|kill|select|sys|sysobjects|syscolumns|table|update)",
                "SQL injection keywords",
                "medium",
                "sql_injection"
            ),
            
            # XSS Attempts
            ThreatIndicator(
                r"<(script|iframe|object|embed|form|input|svg|math|base|link|meta|style)",
                "XSS tag injection",
                "high",
                "xss"
            ),
            ThreatIndicator(
                r"(javascript:|data:text/html|vbscript:|onclick=|onerror=|onload=|eval\(|expression\()",
                "XSS payload attempt",
                "high",
                "xss"
            ),
            
            # Path Traversal
            ThreatIndicator(
                r"(\.\./|\.\.\\|%2e%2e%2f|%252e%252e%252f|\.\.%2f|\.\.%5c)",
                "Path traversal attempt",
                "high",
                "path_traversal"
            ),
            
            # Command Injection
            ThreatIndicator(
                r"(;|\||`|\$\(|\${|&&|\|\||>>|<<)",
                "Command injection characters",
                "medium",
                "command_injection"
            ),
            ThreatIndicator(
                r"(whoami|/etc/passwd|/etc/shadow|cmd\.exe|powershell|wget|curl\s+http)",
                "Command injection payloads",
                "high",
                "command_injection"
            ),
            
            # XXE
            ThreatIndicator(
                r"<!ENTITY.*SYSTEM|<!DOCTYPE.*\[|SYSTEM\s+[\"']file://",
                "XXE injection attempt",
                "high",
                "xxe"
            ),
            
            # LDAP Injection
            ThreatIndicator(
                r"[*()\\|&=]|(objectClass=\*)",
                "LDAP injection attempt",
                "medium",
                "ldap_injection"
            ),
            
            # File Upload
            ThreatIndicator(
                r"\.(php|phtml|php3|php4|php5|phps|asp|aspx|jsp|jspx|cgi|pl|py|sh|bat|exe|dll|scr|vbs|js)$",
                "Malicious file extension",
                "high",
                "malicious_file"
            ),
            
            # Scanner Detection
            ThreatIndicator(
                r"(nikto|nmap|masscan|sqlmap|burp|zap|acunetix|nessus|openvas|metasploit)",
                "Security scanner detected",
                "medium",
                "scanner"
            ),
        ]
    
    def analyze_request(self, 
                       method: str,
                       path: str,
                       headers: Dict[str, str],
                       body: Optional[str] = None) -> List[SecurityEvent]:
        """Analyze request for threats"""
        events = []
        
        # Combine all request data for analysis
        request_data = f"{method} {path} "
        request_data += " ".join([f"{k}: {v}" for k, v in headers.items()])
        if body:
            request_data += f" {body}"
        
        # Check against threat indicators
        for indicator in self.indicators:
            if indicator.matches(request_data):
                events.append(SecurityEvent(
                    timestamp=datetime.utcnow(),
                    event_type=indicator.category,
                    severity=indicator.severity,
                    source_ip=headers.get("X-Real-IP", "unknown"),
                    user_id=None,  # Will be set by caller
                    details={
                        "indicator": indicator.pattern,
                        "description": indicator.description,
                        "matched_in": "request"
                    },
                    raw_request=request_data[:1000]  # Truncate for storage
                ))
        
        return events


class AnomalyDetector:
    """Behavioral anomaly detection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger()
        
        # Thresholds
        self.thresholds = {
            "requests_per_minute": 60,
            "failed_auth_per_hour": 10,
            "unique_paths_per_minute": 20,
            "error_rate_percent": 20,
            "large_file_uploads_per_hour": 5,
            "concurrent_sessions": 5
        }
    
    async def analyze_behavior(self, 
                             identifier: str,
                             identifier_type: str = "ip") -> List[SecurityEvent]:
        """Analyze behavior patterns for anomalies"""
        events = []
        current_time = datetime.utcnow()
        
        # Get behavior metrics
        metrics = await self._get_behavior_metrics(identifier, identifier_type)
        
        # Check request rate anomaly
        if metrics["requests_per_minute"] > self.thresholds["requests_per_minute"]:
            events.append(SecurityEvent(
                timestamp=current_time,
                event_type="rate_anomaly",
                severity="medium",
                source_ip=identifier if identifier_type == "ip" else "unknown",
                user_id=identifier if identifier_type == "user" else None,
                details={
                    "requests_per_minute": metrics["requests_per_minute"],
                    "threshold": self.thresholds["requests_per_minute"]
                }
            ))
        
        # Check failed authentication anomaly
        if metrics["failed_auth_per_hour"] > self.thresholds["failed_auth_per_hour"]:
            events.append(SecurityEvent(
                timestamp=current_time,
                event_type="brute_force_attempt",
                severity="high",
                source_ip=identifier if identifier_type == "ip" else "unknown",
                user_id=identifier if identifier_type == "user" else None,
                details={
                    "failed_attempts": metrics["failed_auth_per_hour"],
                    "threshold": self.thresholds["failed_auth_per_hour"]
                }
            ))
        
        # Check path scanning anomaly
        if metrics["unique_paths_per_minute"] > self.thresholds["unique_paths_per_minute"]:
            events.append(SecurityEvent(
                timestamp=current_time,
                event_type="path_scanning",
                severity="medium",
                source_ip=identifier if identifier_type == "ip" else "unknown",
                user_id=identifier if identifier_type == "user" else None,
                details={
                    "unique_paths": metrics["unique_paths_per_minute"],
                    "threshold": self.thresholds["unique_paths_per_minute"],
                    "paths": metrics["recent_paths"][:10]
                }
            ))
        
        # Check error rate anomaly
        if metrics["error_rate_percent"] > self.thresholds["error_rate_percent"]:
            events.append(SecurityEvent(
                timestamp=current_time,
                event_type="high_error_rate",
                severity="low",
                source_ip=identifier if identifier_type == "ip" else "unknown",
                user_id=identifier if identifier_type == "user" else None,
                details={
                    "error_rate": metrics["error_rate_percent"],
                    "threshold": self.thresholds["error_rate_percent"]
                }
            ))
        
        return events
    
    async def _get_behavior_metrics(self, 
                                  identifier: str,
                                  identifier_type: str) -> Dict[str, Any]:
        """Get behavior metrics from Redis"""
        prefix = f"behavior:{identifier_type}:{identifier}"
        
        # Get various counters
        pipe = self.redis.pipeline()
        pipe.get(f"{prefix}:requests:1min")
        pipe.get(f"{prefix}:failed_auth:1hour")
        pipe.smembers(f"{prefix}:paths:1min")
        pipe.get(f"{prefix}:errors:1min")
        pipe.get(f"{prefix}:total:1min")
        
        results = pipe.execute()
        
        requests_per_minute = int(results[0] or 0)
        failed_auth_per_hour = int(results[1] or 0)
        unique_paths = results[2] or set()
        errors = int(results[3] or 0)
        total = int(results[4] or 1)  # Avoid division by zero
        
        return {
            "requests_per_minute": requests_per_minute,
            "failed_auth_per_hour": failed_auth_per_hour,
            "unique_paths_per_minute": len(unique_paths),
            "error_rate_percent": (errors / total) * 100 if total > 0 else 0,
            "recent_paths": list(unique_paths)[:20]
        }


class IncidentResponseSystem:
    """Automated incident response system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger()
        self.response_actions = self._configure_response_actions()
    
    def _configure_response_actions(self) -> Dict[str, List[str]]:
        """Configure automated response actions"""
        return {
            "sql_injection": ["block_ip", "alert_security", "log_full_request"],
            "xss": ["block_request", "alert_security", "log_full_request"],
            "path_traversal": ["block_ip", "alert_security"],
            "command_injection": ["block_ip", "alert_critical", "log_full_request"],
            "brute_force_attempt": ["temporary_block", "increase_monitoring", "alert_security"],
            "rate_anomaly": ["rate_limit_increase", "monitor_closely"],
            "scanner": ["temporary_block", "honeypot_redirect"],
            "malicious_file": ["block_request", "quarantine_file", "alert_security"]
        }
    
    async def respond_to_incident(self, event: SecurityEvent) -> Dict[str, Any]:
        """Execute automated incident response"""
        start_time = time.time()
        actions_taken = []
        
        # Get response actions for event type
        actions = self.response_actions.get(event.event_type, ["log_only"])
        
        for action in actions:
            try:
                if action == "block_ip":
                    await self._block_ip(event.source_ip, duration=3600)  # 1 hour
                    actions_taken.append("ip_blocked")
                
                elif action == "temporary_block":
                    await self._block_ip(event.source_ip, duration=300)  # 5 minutes
                    actions_taken.append("ip_temporarily_blocked")
                
                elif action == "block_request":
                    # This would be handled at the middleware level
                    actions_taken.append("request_blocked")
                
                elif action == "alert_security":
                    await self._send_security_alert(event, "security")
                    actions_taken.append("security_alerted")
                
                elif action == "alert_critical":
                    await self._send_security_alert(event, "critical")
                    actions_taken.append("critical_alert_sent")
                
                elif action == "log_full_request":
                    await self._log_forensic_data(event)
                    actions_taken.append("forensic_logged")
                
                elif action == "rate_limit_increase":
                    await self._adjust_rate_limit(event.source_ip, multiplier=0.5)
                    actions_taken.append("rate_limit_reduced")
                
                elif action == "honeypot_redirect":
                    await self._mark_for_honeypot(event.source_ip)
                    actions_taken.append("honeypot_marked")
                
                elif action == "quarantine_file":
                    # Would be handled by file processing system
                    actions_taken.append("file_quarantined")
                
            except Exception as e:
                self.logger.error(
                    "incident_response_error",
                    action=action,
                    error=str(e),
                    exc_info=True
                )
        
        # Record metrics
        response_time = time.time() - start_time
        incident_response_time.labels(incident_type=event.event_type).observe(response_time)
        
        # Log incident response
        self.logger.warning(
            "incident_response_completed",
            event_type=event.event_type,
            severity=event.severity,
            actions_taken=actions_taken,
            response_time=response_time
        )
        
        return {
            "event_id": hashlib.sha256(
                f"{event.timestamp}{event.source_ip}{event.event_type}".encode()
            ).hexdigest()[:16],
            "actions_taken": actions_taken,
            "response_time": response_time
        }
    
    async def _block_ip(self, ip_address: str, duration: int):
        """Block IP address"""
        if not ip_address or ip_address == "unknown":
            return
        
        # Add to blocklist with expiration
        self.redis.setex(
            f"blocked_ip:{ip_address}",
            duration,
            json.dumps({
                "blocked_at": datetime.utcnow().isoformat(),
                "duration": duration,
                "reason": "automated_response"
            })
        )
        
        blocked_requests.labels(reason="incident_response", ip_address=ip_address).inc()
    
    async def _send_security_alert(self, event: SecurityEvent, priority: str):
        """Send security alert to team"""
        alert_data = {
            "timestamp": event.timestamp.isoformat(),
            "priority": priority,
            "event_type": event.event_type,
            "severity": event.severity,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "details": event.details,
            "raw_request": event.raw_request
        }
        
        # In production, integrate with:
        # - PagerDuty for critical alerts
        # - Slack for security channel
        # - Email for audit trail
        # - SIEM system for correlation
        
        self.logger.critical(
            "security_alert",
            **alert_data
        )
        
        # Store alert for audit
        self.redis.lpush(
            "security_alerts",
            json.dumps(alert_data)
        )
        self.redis.ltrim("security_alerts", 0, 999)  # Keep last 1000
    
    async def _log_forensic_data(self, event: SecurityEvent):
        """Log detailed forensic data"""
        forensic_data = {
            "event": event.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "forensic_id": hashlib.sha256(
                f"{event.timestamp}{event.source_ip}".encode()
            ).hexdigest()
        }
        
        # Store in forensic log
        self.redis.hset(
            "forensic_log",
            forensic_data["forensic_id"],
            json.dumps(forensic_data)
        )
        
        # Set expiration (30 days)
        self.redis.expire("forensic_log", 2592000)
    
    async def _adjust_rate_limit(self, identifier: str, multiplier: float):
        """Adjust rate limit for identifier"""
        self.redis.setex(
            f"rate_limit_adjustment:{identifier}",
            300,  # 5 minutes
            str(multiplier)
        )
    
    async def _mark_for_honeypot(self, ip_address: str):
        """Mark IP for honeypot redirection"""
        self.redis.setex(
            f"honeypot_redirect:{ip_address}",
            3600,  # 1 hour
            "true"
        )


class SecurityMonitor:
    """Main security monitoring orchestrator"""
    
    def __init__(self):
        self.redis = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        self.threat_intel = ThreatIntelligence()
        self.anomaly_detector = AnomalyDetector(self.redis)
        self.incident_response = IncidentResponseSystem(self.redis)
        self.logger = structlog.get_logger()
        
        # Event queue for processing
        self.event_queue = asyncio.Queue(maxsize=1000)
        
        # Threat scores
        self.threat_scores = defaultdict(lambda: 0)
    
    async def analyze_request(self,
                            method: str,
                            path: str,
                            headers: Dict[str, str],
                            body: Optional[str] = None,
                            user_id: Optional[str] = None) -> bool:
        """
        Analyze incoming request for threats.
        Returns True if request should be allowed, False if blocked.
        """
        source_ip = headers.get("X-Real-IP", headers.get("X-Forwarded-For", "unknown"))
        
        # Check if IP is blocked
        if self.redis.exists(f"blocked_ip:{source_ip}"):
            blocked_requests.labels(reason="ip_blocked", ip_address=source_ip).inc()
            return False
        
        # Threat intelligence analysis
        threat_events = self.threat_intel.analyze_request(method, path, headers, body)
        
        # Update user_id in events
        for event in threat_events:
            event.user_id = user_id
        
        # Anomaly detection
        anomaly_events = await self.anomaly_detector.analyze_behavior(
            source_ip, 
            "ip"
        )
        
        if user_id:
            user_anomalies = await self.anomaly_detector.analyze_behavior(
                user_id,
                "user"
            )
            anomaly_events.extend(user_anomalies)
        
        # Combine all events
        all_events = threat_events + anomaly_events
        
        # Calculate threat score
        threat_score_value = self._calculate_threat_score(all_events)
        self.threat_scores[source_ip] = threat_score_value
        
        # Update metrics
        threat_score.labels(
            identifier_type="ip",
            identifier=source_ip
        ).set(threat_score_value)
        
        # Process high severity events immediately
        block_request = False
        for event in all_events:
            if event.severity in ["high", "critical"]:
                # Queue for incident response
                await self.event_queue.put(event)
                
                # Block critical threats immediately
                if event.severity == "critical":
                    block_request = True
                    blocked_requests.labels(
                        reason=event.event_type,
                        ip_address=source_ip
                    ).inc()
        
        # Block if threat score too high
        if threat_score_value > 80:
            block_request = True
            blocked_requests.labels(
                reason="high_threat_score",
                ip_address=source_ip
            ).inc()
        
        return not block_request
    
    def _calculate_threat_score(self, events: List[SecurityEvent]) -> float:
        """Calculate threat score based on events"""
        score = 0
        severity_scores = {
            "low": 10,
            "medium": 25,
            "high": 50,
            "critical": 100
        }
        
        for event in events:
            score += severity_scores.get(event.severity, 0)
        
        # Cap at 100
        return min(score, 100)
    
    async def process_events(self):
        """Process security events from queue"""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Update metrics
                security_events.labels(
                    event_type=event.event_type,
                    severity=event.severity,
                    action_taken="processing"
                ).inc()
                
                # Execute incident response
                response = await self.incident_response.respond_to_incident(event)
                
                # Log processed event
                self.logger.info(
                    "security_event_processed",
                    event_type=event.event_type,
                    severity=event.severity,
                    response=response
                )
                
            except Exception as e:
                self.logger.error(
                    "event_processing_error",
                    error=str(e),
                    exc_info=True
                )
            
            await asyncio.sleep(0.1)  # Prevent tight loop
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        # Get recent events
        recent_alerts = self.redis.lrange("security_alerts", 0, 99)
        alerts = [json.loads(alert) for alert in recent_alerts]
        
        # Get blocked IPs
        blocked_ips = []
        for key in self.redis.scan_iter("blocked_ip:*"):
            ip = key.split(":")[-1]
            blocked_data = json.loads(self.redis.get(key))
            blocked_ips.append({
                "ip": ip,
                "blocked_at": blocked_data["blocked_at"],
                "duration": blocked_data["duration"]
            })
        
        # Calculate statistics
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in alerts:
            event_counts[alert["event_type"]] += 1
            severity_counts[alert["severity"]] += 1
        
        return {
            "report_time": datetime.utcnow().isoformat(),
            "statistics": {
                "total_alerts": len(alerts),
                "blocked_ips": len(blocked_ips),
                "event_types": dict(event_counts),
                "severity_distribution": dict(severity_counts),
                "top_threat_scores": sorted(
                    [(ip, score) for ip, score in self.threat_scores.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            },
            "recent_alerts": alerts[:20],
            "blocked_ips": blocked_ips[:20],
            "metrics": generate_latest().decode('utf-8')
        }


# Middleware integration
class SecurityMiddleware:
    """FastAPI middleware for security monitoring"""
    
    def __init__(self, app, monitor: SecurityMonitor):
        self.app = app
        self.monitor = monitor
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract request details
            headers = dict(scope["headers"])
            headers = {k.decode(): v.decode() for k, v in headers.items()}
            
            method = scope["method"]
            path = scope["path"]
            
            # Read body if present
            body = None
            if method in ["POST", "PUT", "PATCH"]:
                # This is simplified - in production use proper body reading
                pass
            
            # Analyze request
            user_id = scope.get("user", {}).get("id")
            allowed = await self.monitor.analyze_request(
                method, path, headers, body, user_id
            )
            
            if not allowed:
                # Block request
                response = {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"application/json")]
                }
                await send(response)
                
                body = json.dumps({
                    "error": "Access denied",
                    "reason": "Security policy violation"
                }).encode()
                
                await send({
                    "type": "http.response.body",
                    "body": body
                })
                return
        
        # Continue normal processing
        await self.app(scope, receive, send)


# Example usage
if __name__ == "__main__":
    # Initialize monitoring system
    monitor = SecurityMonitor()
    
    # Start event processor
    asyncio.create_task(monitor.process_events())
    
    # Example: Analyze a suspicious request
    suspicious_headers = {
        "X-Real-IP": "192.168.1.100",
        "User-Agent": "sqlmap/1.0"
    }
    
    async def test_monitoring():
        allowed = await monitor.analyze_request(
            "GET",
            "/api/v1/jobs/1' OR '1'='1",
            suspicious_headers
        )
        print(f"Request allowed: {allowed}")
        
        # Generate report
        report = await monitor.generate_security_report()
        print(f"Security report: {json.dumps(report, indent=2)}")
    
    # Run test
    asyncio.run(test_monitoring())