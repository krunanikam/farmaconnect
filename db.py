import sqlite3

def fetch_and_display_commodities():
    try:
        # Connect to SQLite database
        conn = sqlite3.connect("farmer.db")
        cursor = conn.cursor()

        # Fetch all data from the commodities table
        cursor.execute("SELECT * FROM commodities")

        # Fetch all rows
        rows = cursor.fetchall()

        # Print column headers
        print("ID | Name | Address | Nearby Market | Pincode | Mobile | Commodity | Price | Quantity")
        print("-" * 90)

        # Print each row
        for row in rows:
            print(
                f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | ₹{row[7]} | {row[8]}"
            )

    except sqlite3.Error as e:
        print(f"An error occurred while accessing the database: {e}")

    finally:
        # Close the connection
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

# Run the function
fetch_and_display_commodities()