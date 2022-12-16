from Trade import Trade

def main():
    start = "2021-09-26 00:00:00"
    finish = "2022-09-25 23:00:00"

    trade = Trade(window=240, start=start, finish=finish)
    trade.run_strategy()
    trade.report_results()


if __name__ == "__main__":
    main()