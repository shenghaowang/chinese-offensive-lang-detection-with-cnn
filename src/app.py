import streamlit as st


def main():
    st.write(
        """
    # Chinese Offensive Language Detector
    Hello *world!*
    """
    )

    _ = st.text_area("Enter text", height=275)


if __name__ == "__main__":
    main()
