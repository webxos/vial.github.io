from fastapi import Depends

def check_rbac(user=Depends()):
    return True
