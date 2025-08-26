#!/usr/bin/env python3
"""
Cross-Platform Compatibility Assessment for Photo-Restore ML Stack
Analyzes system compatibility and generates deployment recommendations.
"""

import sys
import platform
import os
import subprocess
import json
from pathlib import Path


def detect_platform():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    release = platform.release()
    
    platform_info = {
        'system': system,
        'machine': machine,
        'release': release,
        'python_version': sys.version,
        'platform_string': platform.platform(),
        'compatibility_tier': 'unknown'
    }
    
    # Determine compatibility tier
    if system == 'linux':
        if machine in ['x86_64', 'amd64']:
            platform_info['compatibility_tier'] = 'tier1_full'
        elif machine in ['aarch64', 'arm64']:
            platform_info['compatibility_tier'] = 'tier1_full'
        elif 'arm' in machine:
            platform_info['compatibility_tier'] = 'tier2_limited'
        else:
            platform_info['compatibility_tier'] = 'tier3_unsupported'
            
    elif system == 'darwin':  # macOS
        if machine in ['x86_64', 'arm64']:
            platform_info['compatibility_tier'] = 'tier2_limited'
        else:
            platform_info['compatibility_tier'] = 'tier3_unsupported'
            
    elif system == 'windows':
        # Check if WSL
        if 'microsoft' in release.lower() or 'wsl' in os.environ.get('WSL_DISTRO_NAME', ''):
            platform_info['compatibility_tier'] = 'tier1_full'
            platform_info['wsl_detected'] = True
        else:
            platform_info['compatibility_tier'] = 'tier3_unsupported'
            platform_info['windows_native'] = True
    
    return platform_info


def check_system_dependencies():
    """Check for required system dependencies."""
    dependencies = {
        'build_tools': {
            'gcc': 'gcc --version',
            'g++': 'g++ --version', 
            'make': 'make --version',
            'cmake': 'cmake --version'
        },
        'python_dev': {
            'python_headers': f'/usr/include/python{sys.version_info.major}.{sys.version_info.minor}',
            'pip': 'pip --version',
            'setuptools': 'python -c "import setuptools; print(setuptools.__version__)"'
        },
        'optional': {
            'git': 'git --version',
            'curl': 'curl --version',
            'wget': 'wget --version'
        }
    }
    
    results = {}
    
    for category, tools in dependencies.items():
        results[category] = {}
        for tool, check_cmd in tools.items():
            try:
                if check_cmd.startswith('/'):
                    # File path check
                    results[category][tool] = {
                        'available': os.path.exists(check_cmd),
                        'type': 'file_check',
                        'path': check_cmd
                    }
                else:
                    # Command check
                    result = subprocess.run(
                        check_cmd.split(), 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    results[category][tool] = {
                        'available': result.returncode == 0,
                        'type': 'command_check',
                        'version': result.stdout.strip().split('\n')[0] if result.returncode == 0 else None,
                        'error': result.stderr if result.returncode != 0 else None
                    }
            except subprocess.TimeoutExpired:
                results[category][tool] = {
                    'available': False,
                    'type': 'timeout',
                    'error': 'Command timeout'
                }
            except FileNotFoundError:
                results[category][tool] = {
                    'available': False,
                    'type': 'not_found',
                    'error': 'Command not found'
                }
            except Exception as e:
                results[category][tool] = {
                    'available': False,
                    'type': 'error',
                    'error': str(e)
                }
    
    return results


def assess_ml_compatibility():
    """Assess ML package compatibility for current platform."""
    compatibility = {
        'pytorch': {
            'cpu_support': True,  # PyTorch supports CPU on all platforms
            'recommended_version': '>=1.9.0,<2.0.0',
            'installation_method': 'pip'
        },
        'basicsr': {
            'native_support': False,
            'requires_compilation': True,
            'success_probability': 'unknown',
            'alternatives': ['manual_implementation', 'docker']
        },
        'realesrgan': {
            'dependency_on_basicsr': True,
            'fallback_available': True
        },
        'gfpgan': {
            'dependency_on_basicsr': True,
            'fallback_available': True
        }
    }
    
    platform_info = detect_platform()
    
    # Adjust compatibility based on platform
    if platform_info['compatibility_tier'] == 'tier1_full':
        compatibility['basicsr']['success_probability'] = 'high'
        compatibility['basicsr']['native_support'] = True
    elif platform_info['compatibility_tier'] == 'tier2_limited':
        compatibility['basicsr']['success_probability'] = 'medium'
        compatibility['pytorch']['installation_method'] = 'conda_preferred'
    else:
        compatibility['basicsr']['success_probability'] = 'low'
        compatibility['basicsr']['recommended_approach'] = 'docker'
    
    return compatibility


def generate_deployment_recommendations():
    """Generate deployment recommendations based on platform assessment."""
    platform_info = detect_platform()
    system_deps = check_system_dependencies()
    ml_compat = assess_ml_compatibility()
    
    recommendations = {
        'platform': platform_info['compatibility_tier'],
        'primary_strategy': 'unknown',
        'fallback_strategies': [],
        'system_preparation': [],
        'warnings': [],
        'confidence': 'unknown'
    }
    
    # Generate recommendations based on compatibility tier
    if platform_info['compatibility_tier'] == 'tier1_full':
        recommendations['primary_strategy'] = 'portable_setup_script'
        recommendations['fallback_strategies'] = ['working_basicsr_setup', 'manual_implementation']
        recommendations['confidence'] = 'high'
        
        # Check for missing dependencies
        if not system_deps['build_tools']['gcc']['available']:
            recommendations['system_preparation'].append('sudo apt install build-essential')
        
        if not system_deps['python_dev']['python_headers']['available']:
            recommendations['system_preparation'].append(f'sudo apt install python{sys.version_info.major}.{sys.version_info.minor}-dev')
            
    elif platform_info['compatibility_tier'] == 'tier2_limited':
        recommendations['primary_strategy'] = 'conda_environment'
        recommendations['fallback_strategies'] = ['docker', 'manual_implementation']
        recommendations['confidence'] = 'medium'
        recommendations['warnings'].append('BasicSR compilation may fail - manual implementation recommended')
        
        if platform_info['system'] == 'darwin':
            recommendations['system_preparation'].append('brew install cmake gcc')
            
    else:
        recommendations['primary_strategy'] = 'docker'
        recommendations['fallback_strategies'] = ['wsl2_if_windows', 'cloud_deployment']
        recommendations['confidence'] = 'low'
        recommendations['warnings'].append('Native installation not recommended - use containerization')
    
    return recommendations


def create_platform_specific_guide():
    """Create platform-specific installation guide."""
    platform_info = detect_platform()
    recommendations = generate_deployment_recommendations()
    
    guide = {
        'platform_detected': platform_info,
        'installation_guide': {
            'primary_method': recommendations['primary_strategy'],
            'steps': [],
            'fallback_options': recommendations['fallback_strategies'],
            'troubleshooting': []
        }
    }
    
    # Generate step-by-step instructions
    if recommendations['primary_strategy'] == 'portable_setup_script':
        guide['installation_guide']['steps'] = [
            'Update system packages',
            'Install build dependencies',
            'Run portable_setup.sh',
            'Verify installation'
        ]
        
    elif recommendations['primary_strategy'] == 'conda_environment':
        guide['installation_guide']['steps'] = [
            'Install Miniconda/Anaconda',
            'Create isolated environment',
            'Install PyTorch via conda',
            'Install remaining packages via pip',
            'Test functionality'
        ]
        
    elif recommendations['primary_strategy'] == 'docker':
        guide['installation_guide']['steps'] = [
            'Install Docker',
            'Build container with working_basicsr_setup.sh',
            'Run container with volume mounts',
            'Expose CLI interface'
        ]
    
    return guide


def main():
    """Run comprehensive cross-platform compatibility assessment."""
    print("🌍 CROSS-PLATFORM COMPATIBILITY ASSESSMENT")
    print("=" * 50)
    
    # Platform detection
    platform_info = detect_platform()
    print(f"📱 Platform: {platform_info['platform_string']}")
    print(f"🏗️  Architecture: {platform_info['machine']}")
    print(f"🎯 Compatibility Tier: {platform_info['compatibility_tier']}")
    
    # System dependencies
    print(f"\n🔧 System Dependencies:")
    system_deps = check_system_dependencies()
    
    for category, tools in system_deps.items():
        print(f"\n  {category.upper()}:")
        for tool, info in tools.items():
            status = "✅" if info['available'] else "❌"
            version = f" ({info.get('version', 'N/A')})" if info['available'] else ""
            print(f"    {status} {tool}{version}")
    
    # ML compatibility
    print(f"\n🤖 ML Package Compatibility:")
    ml_compat = assess_ml_compatibility()
    
    for package, info in ml_compat.items():
        if 'success_probability' in info:
            prob = info['success_probability']
            status = "🟢" if prob == 'high' else "🟡" if prob == 'medium' else "🔴"
            print(f"  {status} {package}: {prob} success probability")
        else:
            print(f"  ℹ️  {package}: Available with fallbacks")
    
    # Deployment recommendations
    print(f"\n🚀 Deployment Recommendations:")
    recommendations = generate_deployment_recommendations()
    
    print(f"  🎯 Primary Strategy: {recommendations['primary_strategy']}")
    print(f"  🔄 Fallback Options: {', '.join(recommendations['fallback_strategies'])}")
    print(f"  📊 Confidence Level: {recommendations['confidence']}")
    
    if recommendations['system_preparation']:
        print(f"  ⚙️  System Preparation:")
        for prep in recommendations['system_preparation']:
            print(f"    - {prep}")
    
    if recommendations['warnings']:
        print(f"  ⚠️  Warnings:")
        for warning in recommendations['warnings']:
            print(f"    - {warning}")
    
    # Generate complete assessment
    assessment = {
        'platform_info': platform_info,
        'system_dependencies': system_deps,
        'ml_compatibility': ml_compat,
        'recommendations': recommendations,
        'platform_guide': create_platform_specific_guide()
    }
    
    # Save assessment
    output_file = Path(f"cross_platform_assessment_{platform_info['system']}_{platform_info['machine']}.json")
    with open(output_file, 'w') as f:
        json.dump(assessment, f, indent=2, default=str)
    
    print(f"\n💾 Complete assessment saved to: {output_file}")
    
    return assessment


if __name__ == "__main__":
    main()