from DataProcess import process_data  # Import processing function
from DataAnalysis import analyze_data  # Import analysis function


def main():
    processed_data = process_data()
    analyze_data(processed_data)


if __name__ == "__main__":
    main()
