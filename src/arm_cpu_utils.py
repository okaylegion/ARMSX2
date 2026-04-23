"""
ARM CPU Detection and Optimization Utilities for ARMSX2

This module provides utilities for detecting ARM CPU features,
microarchitecture information, and optimization capabilities.
"""

import struct
import platform
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class ARMArchitecture(Enum):
    """ARM Architecture versions"""
    ARMV5 = 5
    ARMV6 = 6
    ARMV7 = 7
    ARMV8_32 = 8
    ARMV8_64 = 64
    UNKNOWN = 0


class ARMMicroarchitecture(Enum):
    """Common ARM Microarchitectures"""
    CORTEX_A7 = "Cortex-A7"
    CORTEX_A8 = "Cortex-A8"
    CORTEX_A9 = "Cortex-A9"
    CORTEX_A15 = "Cortex-A15"
    CORTEX_A53 = "Cortex-A53"
    CORTEX_A57 = "Cortex-A57"
    CORTEX_A72 = "Cortex-A72"
    CORTEX_A73 = "Cortex-A73"
    CORTEX_A75 = "Cortex-A75"
    CORTEX_A76 = "Cortex-A76"
    CORTEX_A77 = "Cortex-A77"
    CORTEX_A78 = "Cortex-A78"
    UNKNOWN = "Unknown"


@dataclass
class ARMCPUInfo:
    """Container for ARM CPU information"""
    processor_name: str
    architecture: ARMArchitecture
    microarchitecture: ARMMicroarchitecture
    has_neon: bool
    has_vfpv3: bool
    has_vfpv4: bool
    has_asimd: bool  # NEON for ARMv8
    has_sve: bool    # Scalable Vector Extension
    cache_l1_i: Optional[int]  # L1 instruction cache size
    cache_l1_d: Optional[int]  # L1 data cache size
    cache_l2: Optional[int]    # L2 cache size
    cores: int
    max_frequency_mhz: Optional[int]


class ARMCPUDetector:
    """Detects and analyzes ARM CPU capabilities"""
    
    def __init__(self):
        self.cpu_info: Optional[ARMCPUInfo] = None
        self._detect_cpu()
    
    def _detect_cpu(self) -> None:
        """Detect CPU information from system"""
        try:
            self.cpu_info = self._parse_proc_cpuinfo()
        except Exception as e:
            print(f"Warning: Failed to detect CPU info: {e}")
            self.cpu_info = self._create_default_cpu_info()
    
    def _parse_proc_cpuinfo(self) -> ARMCPUInfo:
        """Parse /proc/cpuinfo to extract CPU information"""
        cpuinfo_data = {}
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key not in cpuinfo_data:
                            cpuinfo_data[key] = value
        except FileNotFoundError:
            raise RuntimeError("Cannot read /proc/cpuinfo")
        
        # Extract processor information
        processor_name = cpuinfo_data.get('Hardware', 'Unknown')
        
        # Detect architecture
        arch = self._detect_architecture(cpuinfo_data)
        
        # Detect microarchitecture
        uarch = self._detect_microarchitecture(cpuinfo_data)
        
        # Detect features
        features = cpuinfo_data.get('Features', '')
        flags = set(features.split())
        
        has_neon = 'neon' in flags
        has_vfpv3 = 'vfpv3' in flags
        has_vfpv4 = 'vfpv4' in flags
        has_asimd = 'asimd' in flags
        has_sve = 'sve' in flags
        
        # Get cache information (simplified)
        cache_info = self._extract_cache_info(cpuinfo_data)
        
        # Count cores
        cores = self._count_cores(cpuinfo_data)
        
        return ARMCPUInfo(
            processor_name=processor_name,
            architecture=arch,
            microarchitecture=uarch,
            has_neon=has_neon,
            has_vfpv3=has_vfpv3,
            has_vfpv4=has_vfpv4,
            has_asimd=has_asimd,
            has_sve=has_sve,
            cache_l1_i=cache_info['l1_i'],
            cache_l1_d=cache_info['l1_d'],
            cache_l2=cache_info['l2'],
            cores=cores,
            max_frequency_mhz=None
        )
    
    def _detect_architecture(self, cpuinfo_data: Dict[str, str]) -> ARMArchitecture:
        """Detect ARM architecture version"""
        cpu_impl = cpuinfo_data.get('CPU implementer', '').lower()
        cpu_arch = cpuinfo_data.get('CPU architecture', '').lower()
        features = cpuinfo_data.get('Features', '').lower()
        
        if 'asimd' in features or 'aarch64' in cpu_arch:
            return ARMArchitecture.ARMV8_64
        elif '8' in cpu_arch:
            return ARMArchitecture.ARMV8_32
        elif '7' in cpu_arch:
            return ARMArchitecture.ARMV7
        elif '6' in cpu_arch:
            return ARMArchitecture.ARMV6
        elif '5' in cpu_arch:
            return ARMArchitecture.ARMV5
        
        return ARMArchitecture.UNKNOWN
    
    def _detect_microarchitecture(self, cpuinfo_data: Dict[str, str]) -> ARMMicroarchitecture:
        """Detect ARM microarchitecture"""
        cpu_part = cpuinfo_data.get('CPU part', '0x0').lower()
        
        # Common CPU part codes
        microarch_map = {
            '0xc07': ARMMicroarchitecture.CORTEX_A7,
            '0xc08': ARMMicroarchitecture.CORTEX_A8,
            '0xc09': ARMMicroarchitecture.CORTEX_A9,
            '0xc0f': ARMMicroarchitecture.CORTEX_A15,
            '0xd03': ARMMicroarchitecture.CORTEX_A53,
            '0xd07': ARMMicroarchitecture.CORTEX_A57,
            '0xd08': ARMMicroarchitecture.CORTEX_A72,
            '0xd09': ARMMicroarchitecture.CORTEX_A73,
            '0xd0a': ARMMicroarchitecture.CORTEX_A75,
            '0xd0b': ARMMicroarchitecture.CORTEX_A76,
            '0xd0d': ARMMicroarchitecture.CORTEX_A77,
            '0xd0e': ARMMicroarchitecture.CORTEX_A78,
        }
        
        return microarch_map.get(cpu_part, ARMMicroarchitecture.UNKNOWN)
    
    def _extract_cache_info(self, cpuinfo_data: Dict[str, str]) -> Dict[str, Optional[int]]:
        """Extract cache information"""
        return {
            'l1_i': None,
            'l1_d': None,
            'l2': None
        }
    
    def _count_cores(self, cpuinfo_data: Dict[str, str]) -> int:
        """Count number of processor cores"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return sum(1 for line in f if line.startswith('processor'))
        except:
            return 1
    
    def _create_default_cpu_info(self) -> ARMCPUInfo:
        """Create default CPU info when detection fails"""
        return ARMCPUInfo(
            processor_name="Unknown ARM Processor",
            architecture=ARMArchitecture.UNKNOWN,
            microarchitecture=ARMMicroarchitecture.UNKNOWN,
            has_neon=False,
            has_vfpv3=False,
            has_vfpv4=False,
            has_asimd=False,
            has_sve=False,
            cache_l1_i=None,
            cache_l1_d=None,
            cache_l2=None,
            cores=1,
            max_frequency_mhz=None
        )
    
    def get_cpu_info(self) -> ARMCPUInfo:
        """Get detected CPU information"""
        return self.cpu_info
    
    def print_cpu_info(self) -> None:
        """Print CPU information in readable format"""
        if not self.cpu_info:
            print("CPU information not available")
            return
        
        info = self.cpu_info
        print(f"Processor: {info.processor_name}")
        print(f"Architecture: ARMv{info.architecture.value}")
        print(f"Microarchitecture: {info.microarchitecture.value}")
        print(f"Cores: {info.cores}")
        print(f"\nSupported Instructions:")
        print(f"  NEON: {info.has_neon}")
        print(f"  VFPv3: {info.has_vfpv3}")
        print(f"  VFPv4: {info.has_vfpv4}")
        print(f"  ASIMD (NEON for ARMv8): {info.has_asimd}")
        print(f"  SVE: {info.has_sve}")
        print(f"\nCache Information:")
        print(f"  L1 Instruction: {info.cache_l1_i} KB" if info.cache_l1_i else "  L1 Instruction: Unknown")
        print(f"  L1 Data: {info.cache_l1_d} KB" if info.cache_l1_d else "  L1 Data: Unknown")
        print(f"  L2: {info.cache_l2} KB" if info.cache_l2 else "  L2: Unknown")


class ARMOptimizationSelector:
    """Selects optimal code paths based on CPU capabilities"""
    
    def __init__(self, cpu_detector: ARMCPUDetector):
        self.detector = cpu_detector
        self.cpu_info = cpu_detector.get_cpu_info()
    
    def use_neon_optimization(self) -> bool:
        """Check if NEON optimizations can be used"""
        return self.cpu_info.has_neon or self.cpu_info.has_asimd
    
    def use_sve_optimization(self) -> bool:
        """Check if SVE optimizations can be used"""
        return self.cpu_info.has_sve
    
    def get_recommended_optimization_flags(self) -> List[str]:
        """Get compiler optimization flags for this CPU"""
        flags = []
        
        if self.cpu_info.architecture == ARMArchitecture.ARMV8_64:
            flags.append("-march=armv8-a")
        elif self.cpu_info.architecture == ARMArchitecture.ARMV7:
            flags.append("-march=armv7-a")
        elif self.cpu_info.architecture == ARMArchitecture.ARMV6:
            flags.append("-march=armv6")
        
        if self.use_neon_optimization():
            flags.append("-mfpu=neon")
        
        if self.use_sve_optimization():
            flags.append("-msve-vector-bits=256")
        
        return flags


# Example usage and testing
if __name__ == "__main__":
    print("ARMSX2 - ARM CPU Detection Utility")
    print("=" * 50)
    
    detector = ARMCPUDetector()
    detector.print_cpu_info()
    
    print("\n" + "=" * 50)
    print("Optimization Recommendations:")
    print("=" * 50)
    
    optimizer = ARMOptimizationSelector(detector)
    print(f"NEON Optimization Available: {optimizer.use_neon_optimization()}")
    print(f"SVE Optimization Available: {optimizer.use_sve_optimization()}")
    print(f"Recommended Compiler Flags: {' '.join(optimizer.get_recommended_optimization_flags())}")
