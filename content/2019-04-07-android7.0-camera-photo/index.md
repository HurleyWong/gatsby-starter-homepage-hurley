---
title: 适配Android7.0以上调取相机拍照并返回照片
tags: [ Android ]
date: 2019-04-07T06:25:44.226Z
path: blog/android-camera-photo
cover: ./android-camera-photo.png
excerpt:
---

Android调取系统相机拍照获取到拍摄照片或从相册中直接选取照片后展示上传是Android开发中很常见的一个功能，实现的思路主要是：

* 自Android 6.0以后对某些涉及用户隐私权限的获取需要动态获取，所以首先是检查权限，如没有权限则动态申请权限，这里我们需要用到的权限是WRITE_EXTERNAL_STORAGE和CAMERA
* 自Android 7.0后系统禁止应用向外部公开file://URI ，因此需要FileProvider来向外界传递URI

* 获取到拍照后的照片，按照现在的手机拍照文件大小来说不做处理直接展示很容易发生OOM，因此这一步需要对图片做压缩处理

### 一、动态申请权限

首先在Mainfest.xml文件中声明权限

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<!-- 因为拍照需要写入文件 所以需要申请读取内存的权限 -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
```

<!-- more -->

点击Button模拟拍照

```java
mBtn.setOnClickListener(view -> {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        // 如果版本大于Android 6.0
        if (! checkPermission()) {
          // 如果没有全部权限，则请求权限
          requestPermissions();
        }
    } else {
        takePhoto();
    }
});

// 检查权限
private boolean checkPermission() {
  	// 是否有权限
  	boolean haveCameraPermission = ContextCompat.checkSelfPermission(mContext, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
  	boolean haveWritePermission = ContextCompat.checkSelfPermission(mContext, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
  	return haveCameraPermission && haveWritePermission;
}

@RequiresApi(api = Build.VERSION_CODES.M)
private void requestPermissions() {
  	requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE);
}

@Override
public void onRequestPermissionsResult(int requestCode, @NonNull  String[] permissions, @NonNull int[] grantResults) {
  	super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  	switch (requestCode) {
        case REQUEST_PERMISSION_CODE:
        		boolean allowAllPermission = false;
        		for (int i = 0; i < grantResults.length; i++) {
              	// 被拒绝授权
              	if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                  	allowAllPermission = false;
                  	break;
                }
              	allowAllPermission = true;
            }
        
        		if (allowAllPermission) {
              	takePhotoOrPickPhoto();
            } else {
              	Toast.makeText(mContext, "该功能需要授权方可使用", Toast.LENGTH_SHORT).show();
            }
        		break;
    }
}
```

在点击拍照按钮后，调用 `ContextCompat.checkSelfPermission( )`方法检查是否有权限，方法返回值为0说明已经授权。没授权的情况下，调用`requestPermissions( )`方法，该方法的第一个参数为一个数组，数组中的值为你要申请的一个或多个权限的值，第二个参数为请求码。

调用`requestPermission( )`方法后我们需要在Activity中重写`onRequestPermissionsResult()`方法，在该方法中会得到回调结果，方法中第一个参数是请求码，第二个参数是我们申请的权限数组，第三个参数数组中每一个值对应申请的每一个权限的返回值，值为0或-1，0代表授权，-1代表拒绝授权。

### 二、FileProvider

在获取所有所需的权限后，我们调取系统相机拍照

```java
private void takePhoto() {
  	// 步骤一：创建存储照片的文件
  	String path = getFilesDir() + File.separator + "images" + File.separator;
    File file = new File(path, "test.jpg");
    if(!file.getParentFile().exists())
        file.getParentFile().mkdirs();
  	if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
    		//步骤二：Android 7.0及以上获取文件 Uri 
      	mUri = FileProvider.getUriForFile(this, "com.example.admin.custmerviewapplication", file);
    } else {
      	// 步骤三：获取文件Uri
     		mUri = Uri.fromFile(file);
    }
  	// 步骤四：调取系统拍照
  	Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
    intent.putExtra(MediaStore.EXTRA_OUTPUT, mUri);
    startActivityForResult(intent, 101);
}
```

在Android 7.0之前我们只需要步骤一、三、四即可调取系统相机拍照，在此之后的话直接这么调取会报`android.os.FileUriExposedException`异常。所以我们需要对Android 7.0及以后的机型适配，采用FileProvider方式。

#### 1. FileProvider是什么

FileProvider是ContentProvider的一个子类，用于应用程序之间私有文件的传递。自Android 7.0后系统禁止应用向外部公开file://URI ，因此需要FileProvider来向外界传递URI，传递的形式是content : //Uri，使用时需要在清单文件中注册。

#### 2. 注册清单文件

```xml
<manifest>
    ...
    <application>
        ...
        <provider
            android:name="android.support.v4.content.FileProvider"
            android:authorities="com.example.admin.custmerviewapplication"
            android:exported="false"
            android:grantUriPermissions="true">
                 <meta-data
                    android:name="android.support.FILE_PROVIDER_PATHS"
                    android:resource="@xml/file_paths" />
        </provider>
        ...
    </application>
</manifest>
```

解释上面provider标签的意思：

**name** 因为我们使用的是V4包下的FileProvider ，所以name的值就是V4包下FileProvider的相对路径值。当然我们也可以自定义类继承于FileProvider，这时候name的值就是我们自定义类的相对路径了

**authorities** 可以理解为标识符，是我们自己自定义的。我们代码中调用getUriForFile方法获取Uri时第二个参数就是这里我们定义的值

**exported** 代表是否可以输出被外部程序使用，填false就行

**android:grantUriPermissions** 是否允许为文件授予临时权限，必须为true

**<meta-data>** 标签里配置的内容是用来指定那个文件夹下的文件是可被共享的

**name** 为固定的值android.support.FILE_PROVIDER_PATHS

**path** 是对应的xml文件路径，@xml/file_paths代表在xml文件下的file_paths文件

#### 3. **指定可共享的文件路径**

在res目录下新建一个xml文件夹，在文件夹下创建一个名为file_paths的xml文件

```xml
<paths xmlns:android="http://schemas.android.com/apk/res/android">
          <!--files-path  相当于 getFilesDir（）-->
    <files-path name="my_images" path="images"/>
          <!--cache-path  相当于 getCacheDir（）-->
    <cache-path name="lalala" path="cache_image"/>
          <!--external-path  相当于 Environment.getExternalStorageDirectory()-->
    < external-path  name="hahaha" path="comeOn"/>
          <!--external-files-path  相当于 getExternalFilesDir("") -->
    <external-files-path name="paly" path="freeSoft"/>
         <!--external-cache-path  相当于 getExternalCacheDir（） --> 
    <external-cache-path  name="lei" path="."/>
    ...
</paths>
```

files-path所代表的路径等于getFilesDir()，打印getFileDir( )它的路径是 /data/user/0/包名/files。什么意思呢，`<files-path name="my_images" path="images"/>`的意思就是`/data/user/0/包名/files + "/files-path标签中path的值/"`路径下的文件是可共享的，在生成Uri时name的值my_images会替代上面的路径`/data/user/0/包名/files / images /`向外暴露。最终的Uri会是`content : //com.example.admin.custmerviewapplication / my_images / test.jpg`

### 三、图片获取并压缩

我们调用`startActivityForResult(intent, 101);`进行拍照，拍照结束后会回调onActivityResult( )方法。

```java
@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
  	super.onActivityResult(requestCode, resultCode, data);
    if (resultCode == RESULT_OK && requestCode == REQUEST_TAKE_PHOTO_CODE) {
      	// 获取系统照片上传
        Bitmap bm = null;
        try {
            bm = getBitmapFormUri(mUri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        mImageView.setImageBitmap(bm);
    }
}
```

通过Uri直接获取图片加载到内存然后显示在ImageView很容易发生OOM，所以还需做进一步的图片压缩。

```java
public Bitmap getBitmapFormUri(Uri uri) throws FileNotFoundException, IOException {
  	InputStream input = getContentResolver().openInputStream(uri);
    // 这一段代码是不加载文件到内存中也得到bitmap的真是宽高，主要是设置inJustDecodeBounds为true
    BitmapFactory.Options onlyBoundsOptions = new BitmapFactory.Options();
    onlyBoundsOptions.inJustDecodeBounds = true;    // 不加载到内存
    onlyBoundsOptions.inDither = true;  // optional
    onlyBoundsOptions.inPreferredConfig = Bitmap.Config.RGB_565;    // optional
    BitmapFactory.decodeStream(input, null, onlyBoundsOptions);
    input.close();
    int originalWidth = onlyBoundsOptions.outWidth;
    int originalHeight = onlyBoundsOptions.outHeight;
    if ((originalWidth == -1) || (originalHeight == -1))
        return null;
    // 图片分辨率以480x800为标准
    float hh = 800f;    // 这里设置高度为800f
    float ww = 480f;    // 这里设置宽度为480f
    // 缩放比，由于是固定比例缩放，只用高或者宽其中一个数据进行计算即可
    int be = 1; // be=1表示不缩放
    if (originalWidth > originalHeight && originalWidth > ww) {
      	// 如果宽度大的话根据宽度固定大小缩放
        be = (int) (originalWidth / ww);
    } else if (originalWidth < originalHeight && originalHeight > hh) {
      // 如果高度高的话根据宽度固定大小缩放
      be = (int) (originalHeight / hh);
    }
    if (be <= 0)
      	be = 1;
    // 比例压缩
    BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
    bitmapOptions.inSampleSize = be;
  	// 设置缩放比例
    bitmapOptions.inDither = true;
    bitmapOptions.inPreferredConfig = Bitmap.Config.RGB_565;
    input = getContentResolver().openInputStream(uri);
    Bitmap bitmap = BitmapFactory.decodeStream(input, null, bitmapOptions);
    input.close();
		return compressImage(bitmap);   // 再进行质量压缩
}

public Bitmap compressImage(Bitmap image) {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    image.compress(Bitmap.CompressFormat.JPEG, 100, baos);
    // 质量压缩方法，这里100表示不压缩，把压缩后的数据存放到baos中
    int options = 100;
    while (baos.toByteArray().length / 1024 > 100) {
      	// 循环判断如果压缩后图片是否大于100kb,大于继续压缩
        baos.reset();//重置baos即清空baos
        // 第一个参数 ：图片格式 ，第二个参数： 图片质量，100为最高，0为最差  ，第三个参数：保存压缩后的数据的流
        image.compress(Bitmap.CompressFormat.JPEG, options, baos);
      	// 这里压缩options，把压缩后的数据存放到baos中
        options -= 10;  // 每次都减少10
        if (options<=0)
            break;
    }
    ByteArrayInputStream isBm = new ByteArrayInputStream(baos.toByteArray());
  	// 把压缩后的数据baos存放到ByteArrayInputStream中
    Bitmap bitmap = BitmapFactory.decodeStream(isBm, null, null);
  	// 把ByteArrayInputStream数据生成图片
    return bitmap;
}
```

压缩的步骤分为两步，第一步是先得到bitmap的真实宽高计算压缩比例，得到压缩比例后进行初步压缩。第二步将初步压缩的bitmap进行质量压缩得到最终的图片。

从相册中选取图片步骤和调取相机拍照的步骤一致，只是创建的intent和在onActivtyResult回调时获取的Uri不同。

```java
// 调用相册
Intent intent = new Intent(Intent.ACTION_PICK,android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_UR）;
startActivityForResult(intent, PICK_IMAGE_CODE);

@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
  	super.onActivityResult(requestCode, resultCode, data);
  	// 获取图片路径
  	if (requestCode == 102 && resultCode == Activity.RESULT_OK && data != null) {
    		// 通过getData获取到Uri
      	mUri = data.getData();
  	}
}
```


