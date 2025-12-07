import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class Transforms:
    """
    For graph datasets, we typically don't apply the same kind of transforms as images.
    This class is kept for API compatibility with the original code structure.
    """

    class MUTAG:
        class GAT:
            # No transforms needed for MUTAG graphs
            # Could add graph augmentation here if desired (e.g., DropEdge)
            train = None
            test = None


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, split_seed=42):
    """
    Create data loaders for graph datasets.

    Args:
        dataset: Dataset name (e.g., 'MUTAG', 'PROTEINS', 'NCI1')
        path: Root directory for dataset storage
        batch_size: Number of graphs per batch
        num_workers: Number of data loading workers
        transform_name: Transform name (for API compatibility, not used for graphs)
        use_test: If True, use actual test split; otherwise use validation split
        shuffle_train: Whether to shuffle training data
        split_seed: Random seed for dataset splitting

    Returns:
        dict: Dictionary with 'train' and 'test' DataLoaders
        int: Number of classes
    """

    # Load dataset from PyTorch Geometric TUDataset
    dataset_path = os.path.join(path, dataset.upper())

    print(f"Loading {dataset} dataset from {dataset_path}...")
    dataset_obj = TUDataset(root=dataset_path, name=dataset.upper())

    # Shuffle dataset with fixed seed for reproducibility
    torch.manual_seed(split_seed)
    dataset_obj = dataset_obj.shuffle()

    # Dataset statistics
    num_graphs = len(dataset_obj)
    num_classes = dataset_obj.num_classes

    # Split dataset: 75% train, 15% val, 10% test
    train_size = int(0.75 * num_graphs)
    val_size = int(0.15 * num_graphs)

    train_dataset = dataset_obj[:train_size]
    val_dataset = dataset_obj[train_size:train_size + val_size]
    test_dataset = dataset_obj[train_size + val_size:]

    if use_test:
        print(f'Using train ({len(train_dataset)}) + test ({len(test_dataset)}) split')
        print('You are going to run models on the test set. Are you sure?')
        eval_dataset = test_dataset
    else:
        print(f'Using train ({len(train_dataset)}) + validation ({len(val_dataset)}) split')
        eval_dataset = val_dataset

    print(f'Dataset: {dataset} | Graphs: {num_graphs} | Classes: {num_classes}')
    print(f'Avg nodes: {sum([data.num_nodes for data in dataset_obj]) / num_graphs:.1f}')
    print(f'Avg edges: {sum([data.num_edges for data in dataset_obj]) / num_graphs:.1f}')

    # Create PyG DataLoaders
    # Note: PyG DataLoader handles variable-size graphs by creating batched "super-graphs"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'test': test_loader,
    }, num_classes
