import pandas as pd
import numpy as np
import os
import curses

# =======================================================
# 1. blank spreadsheet (no fixed template anymore)
# =======================================================

DEFAULT_ROWS = 10
DEFAULT_COLS = 10

def blank_sheet(rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    col_names = [f"Col{i+1}" for i in range(cols)]
    return pd.DataFrame("", index=range(rows), columns=col_names)

Company_template = blank_sheet()

# ==============================================================================================================
# 2 / 3 . Check if company file exists, if not create a new one / Allowing user to input data for the company
# ==============================================================================================================

User_input = input("Enter or Add company:")
spreadsheet_storage = "Analysis (fund)/" + User_input + ".csv"

def Search_validator():
    return(
        len(User_input) <= 5 and
        all(char.isalpha() for char in User_input) and User_input.isupper())

if Search_validator():
    if os.path.exists(spreadsheet_storage):
        print("Company already exists. Loading data...")
        Company_template = pd.read_csv(spreadsheet_storage, index_col=0)
    else:
        print("Creating new company file...")
        Company_template.to_csv(spreadsheet_storage)

# =======================================================
# 4. setting up for curses
# =======================================================

def draw_grid(stdscr, df, current_row, current_col, edit_buffer, editing):
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    cell_width = 12
    header_height = 2
    row_label_width = 5

    # Column headers -- current_row == -1 selects this row for renaming
    for c_idx, col_name in enumerate(df.columns):
        x_pos = row_label_width + 1 + (c_idx * cell_width)
        if current_row == -1 and c_idx == current_col:
            if editing:
                text = edit_buffer
                stdscr.attron(curses.A_UNDERLINE)
            else:
                text = str(col_name)
                stdscr.attron(curses.A_REVERSE)
        else:
            text = str(col_name)
            stdscr.attron(curses.A_BOLD)
        stdscr.addstr(0, x_pos, f"{text[:cell_width-1]:^{cell_width-1}}|")
        stdscr.attroff(curses.A_REVERSE)
        stdscr.attroff(curses.A_UNDERLINE)
        stdscr.attroff(curses.A_BOLD)

    # Data grid
    for r_idx in range(len(df)):
        if r_idx + header_height >= height - 2:
            break

        stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(r_idx + header_height, 0, f"{r_idx:^{row_label_width}}|")
        stdscr.attroff(curses.A_REVERSE)

        for c_idx in range(len(df.columns)):
            x_pos = row_label_width + 1 + (c_idx * cell_width)
            y_pos = r_idx + header_height
            val = str(df.iloc[r_idx, c_idx])

            if r_idx == current_row and c_idx == current_col:
                if editing:
                    display_text = edit_buffer
                    stdscr.attron(curses.A_UNDERLINE)
                else:
                    display_text = val
                    stdscr.attron(curses.A_REVERSE)
            else:
                display_text = val

            formatted_text = f"{display_text[:cell_width-1]:<{cell_width-1}}|"
            stdscr.addstr(y_pos, x_pos, formatted_text)
            stdscr.attroff(curses.A_REVERSE)
            stdscr.attroff(curses.A_UNDERLINE)

    status_y = height - 1
    if editing:
        guide = "[MODE: EDITING] Type value. ENTER to submit, ESC to discard."
    else:
        guide = "[MODE: NAVIGATE] WASD move | ENTER edit | R row | C col | Q save & exit"
    stdscr.addstr(status_y, 0, guide[:width-1], curses.A_DIM)

    stdscr.refresh()


def insert_row(df, at_index):
    """Insert one blank row at at_index, shifting rows below it down."""
    top = df.iloc[:at_index]
    bottom = df.iloc[at_index:]
    blank = pd.DataFrame([[""] * len(df.columns)], columns=df.columns)
    return pd.concat([top, blank, bottom], ignore_index=True)

def delete_row(df, at_index):
    """Remove the row at at_index, unless it's the only row left."""
    if len(df) <= 1:
        return df
    return df.drop(df.index[at_index]).reset_index(drop=True)


def delete_col(df, at_index):
    """Remove the column at at_index, unless it's the only column left."""
    if len(df.columns) <= 1:
        return df
    return df.drop(df.columns[at_index], axis=1)


def insert_col(df, at_index):
    """Insert one blank column at at_index, shifting columns right."""
    name = f"Col{len(df.columns) + 1}"
    df.insert(at_index, name, "")
    return df


def excel_book(stdscr):
    curses.curs_set(1)
    stdscr.keypad(True)

    df = Company_template.copy()

    current_row = 0
    current_col = 0
    editing = False
    edit_buffer = ""

    while True:
        draw_grid(stdscr, df, current_row, current_col, edit_buffer, editing)
        key = stdscr.getch()

        if not editing:
            if key in [curses.KEY_UP, ord('w')]:
                current_row = max(-1, current_row - 1)   # -1 = header row
            elif key in [curses.KEY_DOWN, ord('s')]:
                current_row = min(len(df) - 1, current_row + 1)
            elif key in [curses.KEY_LEFT, ord('a')]:
                current_col = max(0, current_col - 1)
            elif key in [curses.KEY_RIGHT, ord('d')]:
                current_col = min(len(df.columns) - 1, current_col + 1)

            elif key in [10, 13]:
                editing = True
                if current_row == -1:
                    edit_buffer = str(df.columns[current_col])
                else:
                    edit_buffer = str(df.iloc[current_row, current_col])

            elif key == ord('R'):
                at = max(current_row, 0)
                df = insert_row(df, at)

            elif key == ord('C'):
                df = insert_col(df, current_col)

            elif key == ord('r'):
                df = delete_row(df, current_row if current_row >= 0 else 0)
                current_row = min(current_row, len(df) - 1)

            elif key == ord('c'):
                df = delete_col(df, current_col)
                current_col = min(current_col, len(df.columns) - 1)

            elif key in [ord('q'), ord('Q')]:
                df.to_csv(spreadsheet_storage)
                break

        else:
            if key in [10, 13]:
                if current_row == -1:
                    cols = list(df.columns)
                    cols[current_col] = edit_buffer
                    df.columns = cols
                else:
                    try:
                        if '.' in edit_buffer:
                            df.iloc[current_row, current_col] = float(edit_buffer)
                        else:
                            df.iloc[current_row, current_col] = int(edit_buffer)
                    except ValueError:
                        df.iloc[current_row, current_col] = edit_buffer
                editing = False
                edit_buffer = ""
            elif key == 27:
                editing = False
                edit_buffer = ""
            elif key in [curses.KEY_BACKSPACE, 127, 8]:
                edit_buffer = edit_buffer[:-1]
            elif 32 <= key <= 126:
                edit_buffer += chr(key)

    return df


if __name__ == "__main__":
    print("\nTemplate being loaded into editor:")
    print(Company_template)

    input("\nPress ENTER to open the editor...")

    final_df = curses.wrapper(excel_book)

    print("\n--- Final Saved DataFrame Output ---")
    print(final_df)