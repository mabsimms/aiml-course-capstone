import pandas as pd
from capstone.classic import train_classical_model

def test_train_classic_model():
    df_train = pd.DataFrame({
        "text": [
            " a quick snapshot of where we are for the month , as of dec 19 th buyer po # poi dekatherm rate / dth rate _ type daily / total invoice amount pnm 27267 500617 15 , 000 $ 0 . 0900 $ 0 . 0900 total $ 1 , 350 . 00 virginia power 27719 500623 14 , 514 $ 0 . 0500 $ 0 . 2193 daily $ 3 , 182 . 89 cinergy mkt 27467 500621 17 , 600 $ 0 . 1000 $ 0 . 3716 daily $ 6 , 540 . 00 totals 47 , 114 $ 11 , 072 . 89",
            " hi , i have run the var with updated factor loadings for the gold , silver and cocoa bean positions . the biggest change in var from adding this information has been to the cocoa bean position which has increased from approx $ 45 , 900 to $ 506 , 553 . the overal var has not changed by very much as the position file i was sent from andreas had not changed from the 26 th - i have queried this and it will not be a problem to re - run the numbers on monday if i recieve a further file . the var summary for all the metals is as follows : also , after having a conversation with bjorn about stress / scenario analysis i thought i might quickly try to set up a few scenarios to see how sensitive the var is to a position change in aluminium , nickel and copper . i have only applied position increases ( thge direction of the shifts are dependent on the monthly outright position direction ) up until dec 2000 . the shifts and results are given in the following attached spreadsheet . it is interesting that the individual var ' s are particuarly sensitive to increasing the position for nickel . i would like to discuss these results on monday and any further suggestions for senarios would also be gratefully recieved . have a good weekend , kirstee .",
            "free money click here now",
            "claim your reward NOW NOW NOW",
            "the most amazingest offer evars!",
            " start date : 1 / 16 / 02 ; hourahead hour : 16 ; hourahead schedule download failed . manual intervention required ."
        ],
        "feature_a": [10, 12, 14, 7, 5, 1],
        "Label": [0, 0, 1, 1, 1, 1]
    })

    small_grid = { "clf__C": [0.1, 1.0]}

    grid = train_classical_model(
        df_train,
        feature_cols = ["feature_a"],
        param_grid=small_grid,
        n_jobs=1,
        cv=2
    )

    assert "clf__C" in grid.best_params_
    assert grid.best_params_["clf__C"] in [0.1, 1.0]