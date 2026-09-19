from database import create_tables
from register import register_user


def show_menu():
    print()
    print("=" * 45)
    print("     FACE RECOGNITION ATTENDANCE")
    print("=" * 45)
    print("1. Register new user")
    print("2. Start attendance")
    print("3. View users")
    print("4. View attendance")
    print("0. Exit")
    print("=" * 45)


def view_users():
    from database import connect

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, student_id, name
        FROM users
        ORDER BY id
    """)

    users = cursor.fetchall()

    conn.close()

    print()
    print("=" * 50)
    print("REGISTERED USERS")
    print("=" * 50)

    if not users:
        print("No users found.")
        return

    print(
        f"{'ID':<5}"
        f"{'Student ID':<15}"
        f"{'Name':<25}"
    )

    print("-" * 50)

    for user_id, student_id, name in users:
        print(
            f"{user_id:<5}"
            f"{student_id:<15}"
            f"{name:<25}"
        )


def view_attendance():
    from database import connect

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            users.student_id,
            users.name,
            attendance.timestamp
        FROM attendance
        JOIN users
            ON attendance.user_id = users.id
        ORDER BY attendance.timestamp DESC
    """)

    records = cursor.fetchall()

    conn.close()

    print()
    print("=" * 70)
    print("ATTENDANCE HISTORY")
    print("=" * 70)

    if not records:
        print("No attendance records.")
        return

    print(
        f"{'Student ID':<15}"
        f"{'Name':<25}"
        f"{'Time':<25}"
    )

    print("-" * 70)

    for student_id, name, timestamp in records:
        print(
            f"{student_id:<15}"
            f"{name:<25}"
            f"{timestamp:<25}"
        )


def start_attendance():
    try:
        from attendance import run_attendance

        run_attendance()

    except ImportError:
        print()
        print("attendance.py is not ready yet.")


def main():

    # Đảm bảo database tồn tại
    create_tables()

    while True:

        show_menu()

        choice = input(
            "Choose an option: "
        ).strip()

        # ==========================
        # REGISTER
        # ==========================

        if choice == "1":
            register_user()

        # ==========================
        # ATTENDANCE
        # ==========================

        elif choice == "2":
            start_attendance()

        # ==========================
        # USERS
        # ==========================

        elif choice == "3":
            view_users()

        # ==========================
        # ATTENDANCE HISTORY
        # ==========================

        elif choice == "4":
            view_attendance()

        # ==========================
        # EXIT
        # ==========================

        elif choice == "0":

            print()
            print("Program closed.")

            break

        else:

            print()
            print(
                "Invalid option. "
                "Please try again."
            )


if __name__ == "__main__":
    main()