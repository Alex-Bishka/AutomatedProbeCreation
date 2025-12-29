chatMessage = []

MODEL_ID = "llama3.1-8b-it"
feature_set = [
    ### deceptive features from llama3.1-8b-it ###
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "3-resid-post-aa",
    #     "index": 67052,
    #     "strength": TEST_STRENGTH
    # },
    {
        "modelId": MODEL_ID,
        "layer": "11-resid-post-aa",
        "index": 117440,
        "strength": 10
    },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "27-resid-post-aa",
    #     "index": 129666,
    #     "strength": TEST_STRENGTH
    # },

    ### intent features from llama3.1-8b-it ###
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "3-resid-post-aa",
    #     "index": 130524,
    #     "strength": TEST_STRENGTH
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "7-resid-post-aa",
    #     "index": 112494,
    #     "strength": TEST_STRENGTH
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "15-resid-post-aa",
    #     "index": 116826,
    #     "strength": TEST_STRENGTH
    # },

    ### harm features from llama3.1-8b-it ###
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "11-resid-post-aa",
    #     "index": 23072,
    #     "strength": TEST_STRENGTH
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "15-resid-post-aa",
    #     "index": 41483,
    #     "strength": TEST_STRENGTH
    # },

    ### goal features from llama3.1-8b-it ###
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "3-resid-post-aa",
    #     "index": 35038,
    #     "strength": TEST_STRENGTH
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "7-resid-post-aa",
    #     "index": 20536,
    #     "strength": TEST_STRENGTH
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "19-resid-post-aa",
    #     "index": 31143,
    #     "strength": TEST_STRENGTH
    # },

]

# MODEL_ID = "gemma-2-9b-it"
gemma_feature_set = [
    # but what if deception isn't a single feature? what if it's compositional?
    # what if like with hallucinations, we need to combine multiple features?
    # for example, we might first want an intent feature and then a falsehood feature
    {
        "modelId": MODEL_ID,
        "layer": "9-gemmascope-res-131k",
        "index": 50806,  #  concepts related to intent, whether malicious or unintentional
        "strength": 60
    }, 
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "TEST_STRENGTH-gemmascope-res-131k",
    #     "index": 7757,  # evil and badness
    #     "strength": 35
    # }, 
    {
        "modelId": MODEL_ID,
        "layer": "TEST_STRENGTH-gemmascope-res-131k",
        "index": 14522,  #  harm to others
        "strength": 60
    }, 
    
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "TEST_STRENGTH-gemmascope-res-131k",
    #     "index": 74211,  #   themes related to deception and falsehoods
    #     "strength": 30
    # }, 
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "31-gemmascope-res-131k",
    #     "index": 10354,  #   criminal activity and illicit dealings
    #     "strength": 30
    # }, 
    

    

    # singular deception features that don't seem to work too well
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "TEST_STRENGTH-gemmascope-res-131k",
    #     "index": 14987,  # instances of deception or trickery
    #     "strength": 50
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "31-gemmascope-res-131k",
    #     "index": 105168,  # being sneaky or disguised
    #     "strength": 50
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "31-gemmascope-res-131k",
    #     "index": 85701,  # cheating and deception
    #     "strength": 50
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "TEST_STRENGTH-gemmascope-res-131k",
    #     "index": 6614,  # language related to deception and misleading actions
    #     "strength": 5
    # },
    # {
    #     "modelId": MODEL_ID,
    #     "layer": "9-gemmascope-res-131k",
    #     "index": 700TEST_STRENGTH,  # references to deception or falsehood
    #     "strength": 80
    # },
]