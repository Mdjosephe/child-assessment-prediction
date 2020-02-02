import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import json


def preprocessing(df, train_cols=None):

    # Clean dataset
    df = df.copy()
    df['hour'] = df.timestamp.dt.hour
    df['dayofweek'] = df.timestamp.dt.dayofweek
    df['dayofyear'] = df.timestamp.dt.dayofyear
    df_attempts = df[(df.type == 'Assessment') & \
                    ((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)') |
                     (df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)'))].copy()
    df_attempts['success'] = df_attempts.event_data.apply(lambda x: json.loads(x)['correct']) * 1
    passed = df_attempts.groupby(['installation_id', 'game_session']).success.max()\
        .to_frame('passed').reset_index()
    n_attempts = df_attempts.groupby(['installation_id', 'game_session']).size()\
        .to_frame('n_attempts').reset_index()

    # Build features
    # General user features
    X = pd.DataFrame()
    X['n_sessions'] = df.groupby('installation_id').game_session.nunique()
    X['unique_events_per_session'] = df.groupby('installation_id').size()
    X.unique_events_per_session /= X.n_sessions
    game_time = df.groupby(['installation_id', 'game_session']).game_time.max().reset_index()
    X['mean_game_time'] = game_time.groupby('installation_id').game_time.mean()

    # Time user features
    hour = df.groupby(['installation_id', 'game_session']).hour.first().reset_index()
    X['min_hour'] = hour.groupby('installation_id').hour.min()
    X['mean_hour'] = hour.groupby('installation_id').hour.mean()
    X['max_hour'] = hour.groupby('installation_id').hour.max()
    dayofweek_count = df.groupby(['installation_id', 'dayofweek'])\
        .game_session.nunique().to_frame('count').reset_index()
    dayofweek_count.sort_values(['installation_id', 'count'], inplace=True)
    X['most_played_dayofweek'] = dayofweek_count.groupby('installation_id').dayofweek.last()
    X['first_day'] = df.groupby('installation_id').dayofyear.min()
    different_days = df.groupby(['installation_id', 'dayofyear']).size().reset_index()
    X['median_day'] = different_days.groupby('installation_id').dayofyear.median()

    X['hour'] = df.groupby('installation_id').hour.last()
    X['dayofweek'] = df.groupby('installation_id').dayofweek.last()
    X['day'] = df.groupby('installation_id').dayofyear.last()
    X['total_days'] = df.groupby('installation_id').dayofyear.nunique()

    # Predicted game features
    prediction_assessment = df.groupby('installation_id').title.last()\
        .to_frame('last').reset_index()
    prediction_assessment['occurred'] = 1
    prediction_assessment = prediction_assessment\
        .pivot(index='installation_id', columns='last', values='occurred').fillna(0)
    prediction_assessment.columns = ['prediction_assessment_{}'.format(c)
                                     for c in prediction_assessment.columns]
    X = X.merge(prediction_assessment, how='left', left_index=True, right_index=True)

    prediction_world = df.groupby('installation_id').world.last()\
        .to_frame('last').reset_index()
    prediction_world['occurred'] = 1
    prediction_world = prediction_world\
        .pivot(index='installation_id', columns='last', values='occurred').fillna(0)
    prediction_world.columns = ['prediction_world_{}'.format(c)
                                     for c in prediction_world.columns]
    X = X.merge(prediction_world, how='left', left_index=True, right_index=True)

    # Gameplay user features
    type_counts = df.groupby(['installation_id', 'type']).game_session.nunique()\
        .reset_index().pivot(index='installation_id', columns='type', values='game_session').fillna(0)
    type_counts.columns = ['type_{}'.format(c) for c in type_counts.columns]
    X = X.merge(type_counts, how='left', left_index=True, right_index=True)

    world_counts = df.groupby(['installation_id', 'world']).game_session.nunique()\
        .reset_index().pivot(index='installation_id', columns='world', values='game_session').fillna(0)
    world_counts.columns = ['world_{}'.format(c) for c in world_counts.columns]
    X = X.merge(world_counts, how='left', left_index=True, right_index=True)

    event_counts = df.groupby(['installation_id', 'event_code']).event_count.sum()\
        .reset_index().pivot(index='installation_id', columns='event_code', values='event_count').fillna(0)
    event_counts = event_counts.div(X.n_sessions, axis=0)
    event_counts.columns = ['n_events_per_session_{}'.format(c) for c in event_counts.columns]
    X = X.merge(event_counts, how='left', left_index=True, right_index=True)

    # Success user features
    X['mean_passed'] = passed.groupby('installation_id').passed.mean()
    X.mean_passed.fillna(-1, inplace=True)
    X['mean_attempts'] = n_attempts.groupby('installation_id').n_attempts.mean()
    X['max_attempts'] = n_attempts.groupby('installation_id').n_attempts.max()
    X.fillna(0, inplace=True)

    # General user features for each title
    title_sessions = df.groupby(['installation_id', 'title']).game_session.nunique()\
        .reset_index().pivot(index='installation_id', columns='title', values='game_session').fillna(0)
    title_unique_events = df.groupby(['installation_id', 'title']).size().to_frame('size')\
        .reset_index().pivot(index='installation_id', columns='title', values='size')
    title_unique_events  = (title_unique_events / title_sessions).fillna(0)
    title_sessions.columns = ['n_sessions_{}'.format(c) for c in title_sessions.columns]
    title_unique_events.columns = ['unique_events_{}'.format(c) for c in title_unique_events.columns]
    X = X.merge(title_sessions, how='left', left_index=True, right_index=True)
    X = X.merge(title_unique_events, how='left', left_index=True, right_index=True)

    game_time = df.groupby(['installation_id', 'title', 'game_session']).game_time.max().reset_index()
    title_times = game_time.groupby(['installation_id', 'title']).game_time.mean()\
        .reset_index().pivot(index='installation_id', columns='title', values='game_time').fillna(0)
    title_times.columns = ['game_time_{}'.format(c) for c in title_times.columns]
    X = X.merge(title_times, how='left', left_index=True, right_index=True)

    # Success user features for each assessment
    passed = df_attempts.groupby(['installation_id', 'title', 'game_session']).success.max()\
        .to_frame('passed').reset_index()
    n_attempts = df_attempts.groupby(['installation_id', 'title', 'game_session']).size()\
        .to_frame('n_attempts').reset_index()
    title_passed = passed.groupby(['installation_id', 'title']).passed.mean().to_frame('mean')\
        .reset_index().pivot(index='installation_id', columns='title', values='mean')
    title_passed.columns = ['mean_passed_{}'.format(c) for c in title_passed.columns]
    X = X.merge(title_passed, how='left', left_index=True, right_index=True).fillna(-1)

    title_mean_attempts = n_attempts.groupby(['installation_id', 'title'])\
        .n_attempts.mean().to_frame('mean')\
        .reset_index().pivot(index='installation_id', columns='title', values='mean')
    title_mean_attempts.columns = ['mean_attempts_{}'.format(c) for c in title_mean_attempts.columns]
    X = X.merge(title_mean_attempts, how='left', left_index=True, right_index=True).fillna(0)

    title_max_attempts = n_attempts.groupby(['installation_id', 'title'])\
        .n_attempts.max().to_frame('max')\
        .reset_index().pivot(index='installation_id', columns='title', values='max')
    title_max_attempts.columns = ['max_attempts_{}'.format(c) for c in title_max_attempts.columns]
    X = X.merge(title_max_attempts, how='left', left_index=True, right_index=True).fillna(0)
    X['installation_id'] = df.installation_id.unique()

    # Drop columns with only 1 modality
    if train_cols is None:
        X = X.loc[:, X.nunique() != 1]
    else:
        X = X.loc[:, train_cols]

    return X
