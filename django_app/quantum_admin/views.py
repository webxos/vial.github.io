from django.contrib import admin
from django.http import JsonResponse
from server.models.base import QuantumCircuit, Wallet
from server.services.quantum_topology import QuantumTopologyService
from django.views.decorators.csrf import csrf_exempt
import json

@admin.register(QuantumCircuit)
class QuantumCircuitAdmin(admin.ModelAdmin):
    list_display = ('id', 'wallet_id', 'created_at')
    search_fields = ('wallet_id', 'qasm_code')
    
    @csrf_exempt
    def topology_view(self, request):
        """API endpoint for quantum topology management."""
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
                service = QuantumTopologyService()
                result = service.create_topology(data.get('qubits', 8), data.get('gates', []))
                return JsonResponse({"status": "success", "result": result})
            except Exception as e:
                return JsonResponse({"status": "error", "error": str(e)}, status=500)
        return JsonResponse({"status": "error", "message": "Invalid method"}, status=405)

@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = ('id', 'address', 'balance', 'created_at')
    search_fields = ('id', 'address')
