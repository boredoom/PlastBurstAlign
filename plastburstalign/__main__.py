from .plast_burst_align import PlastBurstAlign
from .user_parameters import UserParametersScript


def main():
    params = UserParametersScript()
    burst_align = PlastBurstAlign(params)
    burst_align.execute()
    print("\nend of script\n")


if __name__ == "__main__":
    main()
