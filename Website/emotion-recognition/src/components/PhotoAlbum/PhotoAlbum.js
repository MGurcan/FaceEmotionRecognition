import PhotoAlbum from 'react-photo-album';

const CustomPhotoComponent = ({ photo }) => (
    <div className='flex flex-col justify-content items-center shadow-[0_20px_50px_rgba(8,_112,_184,_0.7)] bg-black text-white rounded-md m-2 p-2'>
        <img className='border-solid border-4 border-gray-600' src={photo.src} alt="" style={{ width: '50%', height: 'auto' }} />
        <img className='border-solid border-4 border-gray-600' src={photo.srcRealPhoto} alt="" style={{ width: '50%', height: 'auto' }} />
        <p>{photo.emotion}</p>
        <p>{photo.time}</p>
    </div>
);

const PhotoAlbumComponent = ({savedImages}) => {
    return (
        <PhotoAlbum
            layout="columns"
            photos={savedImages}
            renderPhoto={(props) => <CustomPhotoComponent {...props} />}
        />
    )
};

export default PhotoAlbumComponent;