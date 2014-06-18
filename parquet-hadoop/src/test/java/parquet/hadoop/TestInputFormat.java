/**
 * Copyright 2012 Twitter, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package parquet.hadoop;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.Test;

import parquet.column.Encoding;
import parquet.hadoop.api.ReadSupport;
import parquet.hadoop.metadata.BlockMetaData;
import parquet.hadoop.metadata.ColumnChunkMetaData;
import parquet.hadoop.metadata.ColumnPath;
import parquet.hadoop.metadata.CompressionCodecName;
import parquet.hadoop.metadata.FileMetaData;
import parquet.hadoop.metadata.ParquetMetadata;
import parquet.schema.MessageType;
import parquet.schema.MessageTypeParser;
import parquet.schema.PrimitiveType.PrimitiveTypeName;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

import static org.junit.Assert.*;
import static org.mockito.Mockito.mock;

public class TestInputFormat {

  @Test
  public void testBlocksToSplits() throws IOException, InterruptedException {
    List<BlockMetaData> blocks = new ArrayList<BlockMetaData>();
    for (int i = 0; i < 10; i++) {
      blocks.add(newBlock(i * 10));
    }
    BlockLocation[] hdfsBlocks = new BlockLocation[] {
        new BlockLocation(new String[0], new String[] { "foo0.datanode", "bar0.datanode"}, 0, 50),
        new BlockLocation(new String[0], new String[] { "foo1.datanode", "bar1.datanode"}, 50, 50)
    };
    FileStatus fileStatus = new FileStatus(100, false, 2, 50, 0, new Path("hdfs://foo.namenode:1234/bar"));
    MessageType schema = MessageTypeParser.parseMessageType("message doc { required binary foo; }");
    FileMetaData fileMetaData = new FileMetaData(schema, new HashMap<String, String>(), "parquet-mr");
    @SuppressWarnings("serial")
    List<ParquetInputSplit> splits = ParquetInputFormat.generateSplits(
        blocks, hdfsBlocks, fileStatus, fileMetaData, ReadSupport.class, schema.toString(), new HashMap<String, String>() {{put("specific", "foo");}});
    assertEquals(splits.toString().replaceAll("([{])", "$0\n").replaceAll("([}])", "\n$0"), 2, splits.size());
    for (int i = 0; i < splits.size(); i++) {
      ParquetInputSplit parquetInputSplit = splits.get(i);
      assertEquals(5, parquetInputSplit.getBlocks().size());
      assertEquals(2, parquetInputSplit.getLocations().length);
      assertEquals("[foo" + i + ".datanode, bar" + i + ".datanode]", Arrays.toString(parquetInputSplit.getLocations()));
      assertEquals(10, parquetInputSplit.getLength());
      assertEquals("foo", parquetInputSplit.getReadSupportMetadata().get("specific"));
    }
  }

  @Test
  public void testFooterCacheEntryIsCurrent() throws IOException, InterruptedException {
    File tempFile = getTempFile();
    FileSystem fs = FileSystem.getLocal(new Configuration());
    ParquetInputFormat.FootersCacheEntry cacheEntry = getDummyCacheEntry(tempFile, fs);

    // wait one second and then access the file to change the access time (we have to wait at least a second to make
    // sure the underlying system can register the difference.
    Thread.sleep(1000);
    BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(tempFile.getPath()))));
    while (br.readLine() != null) { }
    assertTrue(cacheEntry.isCurrent());

    assertTrue(tempFile.setLastModified(tempFile.lastModified() + 5000));
    assertFalse(cacheEntry.isCurrent());
  }

  @Test
  public void testFooterCacheEntryDeleted() throws IOException {
    File tempFile = getTempFile();
    FileSystem fs = FileSystem.getLocal(new Configuration());
    ParquetInputFormat.FootersCacheEntry cacheEntry = getDummyCacheEntry(tempFile, fs);

    assertTrue(tempFile.delete());
    assertFalse(cacheEntry.isCurrent());
  }

  @Test
  public void testFooterCacheEntryIsNewer() throws IOException {
    File tempFile = getTempFile();
    FileSystem fs = FileSystem.getLocal(new Configuration());
    ParquetInputFormat.FootersCacheEntry cacheEntry = getDummyCacheEntry(tempFile, fs);

    assertTrue(cacheEntry.isNewerThan(null));
    assertFalse(cacheEntry.isNewerThan(cacheEntry));

    assertTrue(tempFile.setLastModified(tempFile.lastModified() + 5000));
    ParquetInputFormat.FootersCacheEntry newerCacheEntry = getDummyCacheEntry(tempFile, fs);

    assertTrue(newerCacheEntry.isNewerThan(cacheEntry));
    assertFalse(cacheEntry.isNewerThan(newerCacheEntry));
  }

  private File getTempFile() throws IOException {
    File tempFile = File.createTempFile("footer_", ".txt");
    tempFile.deleteOnExit();
    return tempFile;
  }

  private ParquetInputFormat.FootersCacheEntry getDummyCacheEntry(File file, FileSystem fs) throws IOException {
    Path path = new Path(file.getPath());
    FileStatus status = fs.getFileStatus(path);
    ParquetMetadata mockMetadata = mock(ParquetMetadata.class);
    ParquetInputFormat.FootersCacheEntry cacheEntry = new ParquetInputFormat.FootersCacheEntry(status, new Footer(path, mockMetadata));
    assertTrue(cacheEntry.isCurrent());
    return cacheEntry;
  }


  private BlockMetaData newBlock(long start) {
    BlockMetaData blockMetaData = new BlockMetaData();
    ColumnChunkMetaData column = ColumnChunkMetaData.get(
        ColumnPath.get("foo"), PrimitiveTypeName.BINARY, CompressionCodecName.GZIP, new HashSet<Encoding>(Arrays.asList(Encoding.PLAIN)),
        start, 0l, 0l, 2l, 0l);
    blockMetaData.addColumn(column);
    return blockMetaData;
  }
}
