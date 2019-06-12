# Task: Graph

# DD
#python -m train --datadir=data --task graph --bmname=DD --assign-ratio=0.3 --cuda=0 --max-nodes=500 --num-classes=2 --epochs=1000

# COLLAB
#python -m train --datadir=data --task graph --bmname=COLLAB --assign-ratio=0.3 --cuda=0 --max-nodes=150 --num-classes=3 --epochs=1000

# PROTEINS
#python -m train --datadir=data --task graph --bmname=PROTEINS --assign-ratio=0.1 --cuda=0 --max-nodes=100 --num-classes=1 --epochs=100

# NCI1
#python -m train --datadir=data --task graph --bmname=NCI1 --assign-ratio=0.3 --cuda=0 --max-nodes=150 --num-classes=2 --epochs=100




# Task: Node

# cora
#python -m train --datadir=data --task node --bmname=cora --assign-ratio=0.1 --cuda=0 --epochs=100

# citeseer
#python -m train --datadir=data --task node --bmname=citeseer --assign-ratio=0.1 --cuda=0 --epochs=100 

# pubmed
python -m train --datadir=data --task node --bmname=pubmed --assign-ratio=0.1 --cuda=0 --epochs=100 --hidden-dim=64 --output-dim=64