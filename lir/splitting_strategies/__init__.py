from lir.splitting_strategies.auto import AutoCrossValidation, AutoTrainTestSplit
from lir.splitting_strategies.labels import CrossValidation, TrainTestSplit
from lir.splitting_strategies.leave_one_category_out import LeaveOneCategoryOut
from lir.splitting_strategies.pairs import PairsTrainTestSplit
from lir.splitting_strategies.predefined import PredefinedCrossValidation, PredefinedTrainTestSplit, RoleAssignment
from lir.splitting_strategies.sources import (
    LeaveOneSourceOut,
    LeaveTwoSourcesOut,
    SourcesCrossValidation,
    SourcesTrainTestSplit,
)


__all__ = [
    'TrainTestSplit',
    'CrossValidation',
    'PairsTrainTestSplit',
    'RoleAssignment',
    'PredefinedTrainTestSplit',
    'PredefinedCrossValidation',
    'SourcesTrainTestSplit',
    'SourcesCrossValidation',
    'LeaveOneSourceOut',
    'LeaveTwoSourcesOut',
    'AutoTrainTestSplit',
    'AutoCrossValidation',
    'LeaveOneCategoryOut',
]
