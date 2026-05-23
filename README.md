# boloMarwadi

this is just a simple fun/creative project which i thought would be cool to build, as i'm not able to find any native speakers of my language at the place i currently live in. this was mostly vibe coded with the help of claude code on a weekend while i was having my breakfast, so please don't get pissed off if this doesnt work in the way indented on your local system (it works perfectly fine on my m4 :)). 

this is the directory structure:
├── assets
│   ├── chillGuy_nobg.png
│   └── rajBg.png
├── config.yml
├── index.html
├── LICENSE
├── llm.py
├── main.py
├── memory
│   ├── MEMORY.md
│   └── user_hardware.md
├── models
│   ├── llm
│   │   └── Qwen2.5-7B-Instruct-4bit
│   │       ├── added_tokens.json
│   │       ├── config.json
│   │       ├── merges.txt
│   │       ├── model.safetensors
│   │       ├── model.safetensors.index.json
│   │       ├── README.md
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       └── vocab.json
│   ├── stt
│   │   └── Voxtral-Mini-4B-Realtime-2602-4bit
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── model.safetensors.index.json
│   │       ├── README.md
│   │       └── tekken.json
│   └── tts
│       └── raj
│           ├── fastpitch
│           │   ├── best_model.pth
│           │   ├── config.json
│           │   └── speakers.pth
│           └── hifigan
│               ├── best_model.pth
│               └── config.json
├── prompts
│   └── marwadi_system.md
├── README.md
├── requirements.txt
├── stt.py
├── track_metrics.py
└── tts.py