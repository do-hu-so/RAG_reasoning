"""
Module quản lý đường dẫn cho hệ thống.

Module này chịu trách nhiệm:
- Thêm đường dẫn thủ công vào sys.path
- Đảm bảo các module có thể import được
- Quản lý Python path cho project

Author: Physics Problem Solving System Team
Version: 1.0.0
"""

import sys
import os

def add():
    """
    Thêm đường dẫn thủ công vào sys.path.
    
    Thêm đường dẫn của thư mục cha vào sys.path để
    đảm bảo các module có thể import được.
    
    Returns:
        None
    """
    return sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))