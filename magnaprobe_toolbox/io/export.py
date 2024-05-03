from magnaprobe_toolbox.io import header_order
def data(raw_df, out_fp, display=True, header_order=header_order, drop_header=True):
    """
    :param raw_df: pd.DataFrame()
        Dataframe containing the data to export
    :param out_fp: string
        Output filename.
    :param header_order: list of string
        List containing the ordered headers
    :param drop_header: boolean
        If True, drop any headers not in the list.
        If False, append existing headers not in the ordered header list after the header list.
    :return:
    """
    raw_df.columns = [c[0].upper() + c[1:] for c in raw_df.columns]
    if drop_header:
        header_order = [col for col in header_order if col in raw_df.columns]
    else:
        header_order = [col for col in header_order if col in raw_df.columns] + [col for col in raw_df.columns if col not in header_order]
    raw_df = raw_df[header_order]
    raw_df.to_csv(out_fp, index=False)
    if display:
        print(out_fp)
