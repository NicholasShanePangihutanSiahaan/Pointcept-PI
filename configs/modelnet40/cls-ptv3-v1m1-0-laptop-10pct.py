_base_ = ["./cls-ptv3-v1m1-0-laptop.py"]

# Fast, reproducible experiment configuration for architecture exploration.
# Generate this split first with:
# python tools/create_modelnet_subset.py --ratio 0.10 --seed 42

# In Pointcept, train.loop is computed as epoch // eval_epoch. Keeping these
# equal prevents each trainer epoch from traversing the dataset multiple times.
epoch = 100
eval_epoch = 100

batch_size = 4
batch_size_val = 8
num_worker = 2
enable_wandb = False

data = dict(
    train=dict(
        split="train10",
        num_points=1024,
    ),
    # Keep the complete official test split for an unbiased validation result.
    val=dict(
        split="test",
        num_points=1024,
    ),
    test=dict(
        split="test",
        num_points=1024,
    ),
)

# Ten votes are adequate during rapid experiments. Increase to 100 only for
# the final reported evaluation if time permits.
test = dict(type="ClsVotingTester", num_repeat=10)
