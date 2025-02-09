"""
Agentic RAGシステムの前処理モジュール

PyMuPDF（ChatGPT o3-mini-high）を利用

以下の手順でデータを処理:
1. 質問データ(query.csv)の読み込みと整形
2. 参照元データ(documents.zip)のPDFファイルの抽出とテキスト変換
3. 検証用データ(validation.zip)の展開と前処理
4. 評価用プログラムの準備
"""
import os
import glob
import fitz  # PyMuPDF
from logging import basicConfig, getLogger, StreamHandler, FileHandler,DEBUG, INFO

PDF_FILE_PATH = "./input_data/documents/"
MD_FOLDER_PATH = "./preprocessed_data/documents/"

# ログ設定
basicConfig(level=DEBUG, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s")
Logger = getLogger(__name__)
Logger.setLevel(DEBUG)
# ターミナルへの出力設定
st_handler = StreamHandler()
st_handler.setLevel(INFO)
# ファイルへの出力設定
os.makedirs("./log", exist_ok=True)
fl_handler = FileHandler(filename="./log/RAG.log", encoding="utf-8")
fl_handler.setLevel(DEBUG)
# インスタンス化したハンドラをそれぞれロガーßに渡す
Logger.addHandler(st_handler)
Logger.addHandler(fl_handler)

class PDFToMarkdownConverter:
    """
    １つのPDFファイルを読み込み、ページごとに見開き（2カラム）か単一ページかを判別しながら
    Markdown形式に変換して出力するクラス
    """
    def __init__(self, pdf_file_path, md_folder_path):
        self.pdf_file_path = pdf_file_path
        self.md_folder_path = md_folder_path
        try:
            self.doc = fitz.open(pdf_file_path)
            Logger.info(f"Opened PDF: {pdf_file_path}")
        except Exception as e:
            Logger.error(f"Failed to open PDF {pdf_file_path}: {e}")
            raise e
        self.md_text = ""
        self.page_number = 1


    def extract_blocks(self, page):
        """
        PDFの1ページからテキストブロック（文章のかたまり）を抽出するメソッド
        この関数では、1つのページから「文章のかたまり(ブロック)」だけを取り出す。

        引数:
            page：1ページ分のPDFデータ

        返却値:
            text_blocks：テキストブロックのリスト
        
        各ブロックは以下の7つの情報を持つタプルとして返される:
        - x0, y0: ブロックの左上の座標 (ページ内での位置を表す)
        - x1, y1: ブロックの右下の座標 (ブロックの大きさが分かる)
        - text: 実際のテキスト内容 (ブロック内の文章)
        - block_no: ブロック番号 (ページ内で何番目のブロックか)
        - block_type: ブロックの種類
            - 0の場合: テキスト(文章)
            - 1の場合: 画像やその他の要素
        """
        # PyMuPDFライブラリのget_text()メソッドを使って、ページ内の要素をブロック単位で取得
        # "blocks"を指定することで、文章のかたまりごとに分けて取得できる
        blocks = page.get_text("blocks")

        # デバッグ用に、取得したブロックの内容をログに出力
        Logger.debug(f"blocks:\n{blocks}")

        # b[4]: ブロック内のテキスト内容 → 空でない
        # b[5]: ブロック番号 (ページ内で何番目のブロックか) → ブロック番号が付いているもの
        # b[6]: ブロックタイプ (0:テキスト, 1:画像など) → テキスト(文章)であるもの
        text_blocks = [b for b in blocks if b[5] is not None and b[6] == 0 and b[4].strip()]
        # text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        Logger.debug(f"{page}ページから{len(text_blocks)}個のテキストブロックを抽出")

        return text_blocks


    def classify_columns(self, blocks, page_width):
        """
        PDFページ内のテキストブロックを左右のカラムに分類するメソッド
        
        引数:
            blocks: PDFから抽出したテキストブロックのリスト
            page_width: PDFページの幅
            
        返却値:
            left_blocks: 左カラムに属するブロックのリスト
            right_blocks: 右カラムに属するブロックのリスト
            
        処理の流れ:
        1. 空の左右ブロックリストを用意
        2. 各ブロックに対して:
            - ブロックの左端(x0)と右端(x1)から中心位置を計算
            - ページの中心と比較して左右どちらかに振り分け
        3. 左右のブロックリストを返却
        """
        # 左右のカラムを格納するための空リストを初期化
        left_blocks = []
        right_blocks = []
        
        # 各ブロックを走査して左右に振り分け
        for block in blocks:
            x0, y0, x1, y1, text, _, _ = block
            
            # ブロックの中心位置を計算 (左端x0と右端x1の平均)
            center = (x0 + x1) / 2
            
            # ブロックの中心位置がページの中心より左にあれば左カラム、
            # そうでなければ右カラムに追加
            if center < page_width / 2:
                left_blocks.append(block)
            else:
                right_blocks.append(block)

        Logger.debug(f"page_width:{page_width}")
        Logger.debug(f"left_blocks:{left_blocks}")
        Logger.debug(f"right_blocks:{right_blocks}")

        return left_blocks, right_blocks


    def sort_blocks_reading_order(self, blocks):
        """
        PDFから抽出したテキストブロックを、日本語の自然な読み方順に並べ替えるメソッド
        
        引数:
            blocks: PDFから抽出したテキストブロックのリスト
            
        返却値:
            並べ替えたブロックのリスト
            
        処理の流れ:
        1. sorted()関数を使ってブロックを並べ替え
        2. key=lambda b: (b[1], b[0]) の意味:
           - b[1]はブロックのy座標（縦位置）で、これを第一優先で比較
           - b[0]はブロックのx座標（横位置）で、これを第二優先で比較
        """
        return sorted(blocks, key=lambda b: (b[1], b[0]))


    def join_block_texts(self, blocks):
        """
        各ブロックのテキストを改行区切りで連結する
        """
        texts = [b[4].strip() for b in blocks if b[4].strip()]
        return "\n".join(texts)


    def process(self):
        """
        PDF全ページを処理して Markdown 形式のテキストを生成し、フォルダにファイル名を .md に変更して出力する
        
        このメソッドの処理の流れ:
        1. PDFの各ページを順番に処理
        2. ページごとにテキストブロックを抽出して左右のカラムに分類
        3. 見開きページか単一ページかを判定
        4. テキストを Markdown 形式に変換
        5. 最後に .md ファイルとして保存
        """
        for page in self.doc:
            # ページの幅を取得
            width = page.rect.width
            # ページからテキストブロックを抽出
            blocks = self.extract_blocks(page)
            left_blocks, right_blocks = self.classify_columns(blocks, width)

            # 見開きページかどうかを判定
            # 左右それぞれに2つ以上のテキストブロックがあれば見開きページと判断
            if len(left_blocks) >= 2 and len(right_blocks) >= 2:
                # 見開きページの場合の処理
                
                # 左右のブロックを読み順に並び替え
                left_blocks = self.sort_blocks_reading_order(left_blocks)
                right_blocks = self.sort_blocks_reading_order(right_blocks)
                
                # 左右それぞれのテキストを連結
                left_text = self.join_block_texts(left_blocks)
                right_text = self.join_block_texts(right_blocks)
                
                # Markdown 形式で追加
                self.md_text += f"## ページ {self.page_number}\n\n{left_text}\n\n"
                self.page_number += 1
                self.md_text += f"## ページ {self.page_number}\n\n{right_text}\n\n"
                self.page_number += 1
            else:
                # 単一ページの場合の処理
                # ページ全体のテキストを取得して追加
                full_text = page.get_text().strip()
                self.md_text += f"## ページ {self.page_number}\n\n{full_text}\n\n"
                self.page_number += 1

        base_name = os.path.splitext(os.path.basename(self.pdf_file_path))[0]  # 元のPDFファイル名
        preprocessed_path = os.path.join(self.md_folder_path, base_name + ".md") # 前処理後のデータのファイル名
        
        # ファイルへの書き込みを試行
        try:
            with open(preprocessed_path, "w", encoding="utf-8") as f:
                f.write(self.md_text)
            Logger.info(f"書き込み成功：{preprocessed_path}")
        except Exception as e:
            Logger.error(f"書き込み失敗：{preprocessed_path}: {e}")
            raise e


class FolderPDFConverter:
    """
    指定フォルダ内のすべてのPDFファイルに対してPDFToMarkdownConverter を実行するクラス
    """
    def __init__(self, input_folder_path, output_folder_path):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

    def setup_directories(self):
        """作業用ディレクトリの作成"""
        os.makedirs(self.output_folder_path, exist_ok=True)

    def process_all(self):
        """
        フォルダ内のPDFをすべて処理
        """
        pdf_files = glob.glob(os.path.join(self.input_folder_path, "*.pdf"))
        for pdf_file_path in pdf_files:
            Logger.info(f"============================================================")
            Logger.info(f"処理中のファイル: {pdf_file_path}")
            converter = PDFToMarkdownConverter(pdf_file_path, self.output_folder_path)
            converter.process()


if __name__ == "__main__":
    folder_converter = FolderPDFConverter(PDF_FILE_PATH, MD_FOLDER_PATH)
    folder_converter.setup_directories()
    folder_converter.process_all()