
gpu_name2op_build_map = {
    'V100': 'build_V100',
    '6000': 'build_Q6000',
    '2080': 'build_2080',
    'K80': 'build_K80',
    'P100': 'build_P100',
    'A100': 'build_A100',
}

forbid_guest = [
    "g045",
    "g048",
    "g047",
    "g095",
    "g087",
    "g043",
    "g058",
    "g096",
    "g093",
    "g068",
    "g069",
    "g014",
    
    "g121",
    "g122",
    "g123",
    
    "g128",
    "g101",
]


# g146, NVIDIA A100-SXM-80GB, AttributeError: 'PickableInferenceSession' object has no attribute '_providers'


AVAILABLE_CLUSTER_GPU={
    '0':"Tesla P40",
    '1':"Tesla K80",
    '2':"Quadro RTX 6000",#24220MiB，23GB
    '3':"GeForce RTX 2080 Ti",
    '4':"Tesla P100-PCIE-16GB",
    '5':"Tesla V100-SXM2-32GB",
    '6':"NVIDIA A100-SXM4-40GB",
    '7':"NVIDIA A100-SXM-80GB",#81251MiB
    '9':"NVIDIA A100-SXM4-80GB"
}

TASK_SETTING=dict(
    defaults=dict(
        gpu_type=list(AVAILABLE_CLUSTER_GPU.keys()),
        mem_num=1024*10,
        cpu_num=1,
        
        extra=[]
    ),
    task_smplifyx_tracker=dict(
        gpu_type=['7','9'],
        mem_num=1024*40,
        cpu_num=4,
        
        extra=[
            "'requirements = TARGET.CUDACapability>=10.0'",
            "'requirements = TARGET.CUDAGlobalMemoryMb > 50000'"
        ],
    ),
)